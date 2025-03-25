# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import io
import os
import sys
import base64
import json
import tqdm
import logging
import tempfile
import traceback
from collections import namedtuple

sys.path.append(os.path.dirname(__file__))
#print(sys.path)

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from video_loader import get_video_loader
from oss_utils import get_oss_client, get_oss_query_for, oss_ingest_batch
from search_utils import find_clusters, parse_opensearch_results
from processing_funcs import get_image_embeddings, get_text_embeddings, GaussianSmoothingOverTime


# This code will be loaded on each worker separately..
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define model container bundle
ModelContainer = namedtuple(
    'ModelContainer',
    ['model_text', 'model_image', 'processor_text', 'processor_image', 'smoothing']
)


############################ Load/Setup Config #########################
try:
    CFG = OmegaConf.load(os.path.join(os.environ.get('SAGEMAKER_SUBMIT_DIRECTORY', ''), os.environ['CONFIG_FILE']))
    CFG.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(">>> Config loaded!")
except:
    logger.info(">>> Failed to load config!")
    print(traceback.format_exc())


############################ Container Funcs ###########################
def model_fn(model_dir):

    logger.info(">>> Requested model: '%s'.." % CFG.model_name)

    if 'siglip' in CFG.model_name.lower():
        from transformers import SiglipImageProcessor, SiglipTokenizer, SiglipVisionModel, SiglipTextModel

        processor_image = SiglipImageProcessor.from_pretrained(CFG.model_name)
        processor_text = SiglipTokenizer.from_pretrained(CFG.model_name)

        model_image = SiglipVisionModel.from_pretrained(CFG.model_name).to(CFG.device)
        model_text = SiglipTextModel.from_pretrained(CFG.model_name).to(CFG.device)

    elif 'clip' in CFG.model_name.lower():
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizerFast, CLIPTextModelWithProjection

        processor_image = CLIPImageProcessor.from_pretrained(CFG.model_name)
        processor_text = CLIPTokenizerFast.from_pretrained(CFG.model_name)

        model_image = CLIPVisionModelWithProjection.from_pretrained(CFG.model_name).to(CFG.device)
        model_text = CLIPTextModelWithProjection.from_pretrained(CFG.model_name).to(CFG.device)
    else:
        raise NotImplementedError("Reguested model '%s' is not supported.." % CFG.model_name)

    smoothing = GaussianSmoothingOverTime(
        size=CFG.smoothing.kernel_size,
        sigma=CFG.smoothing.sigma,
    ).to(CFG.device)

    logger.info(">>> Model loaded!")

    return ModelContainer(
        model_text=model_text,
        model_image=model_image,
        processor_text=processor_text,
        processor_image=processor_image,
        smoothing=smoothing
    )


def transform_fn(model, request_body, content_type, accept, context):
    logger.info(f">>> Content type received: '{content_type}'..")

    request_header = context.get_all_request_header(0)
    print(">>> [DEBUG] request_header:", request_header)

    try:
        if content_type == 'video/mp4':
            video_name = request_header['X-Amzn-SageMaker-Custom-Attributes']
            print(">>> [DEBUG] video_name:", video_name)
            return embed_and_index_video(model, request_body, video_name)

        elif content_type == 'application/json':
            logger.info(f">>> Content: '{request_body}'..")
            json_body = json.loads(request_body)

            return search_query(
                model=model,
                image=json_body['query'].get('image', None),
                text=json_body['query'].get('text', None),
                search_size=json_body['search_args']['size'],
                k=json_body['search_args']['k'],
                time_offset=json_body['search_args']['time_offset'],
                search_name=json_body['search_args'].get('name', None)
            )

        else:
            return json.dumps({'Failed': 'Unsupported content type'})

    except Exception:
        return traceback.format_exc()


########################### Processing Funcs ###########################
def embed_and_index_video(model, video_bytes, video_name, do_index=True):
    padding = CFG.smoothing.kernel_size // 2 if CFG.smoothing.enabled else 0

    with io.BytesIO(video_bytes) as f:
        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        tmp_file.write(f.read())

    logger.info(f">>> Video file staged at: '{tmp_file.name}'..")

    video_loader = get_video_loader(
        video_path=tmp_file.name,
        sampling_rate=CFG.video_decoder.sampling_rate,
        batch_size=CFG.video_decoder.batch_size,
        padding=padding
    )

    if do_index:
        logger.info(f">>> Indexing video embeddings to AOSS: {CFG.opensearch.collection_id}.{CFG.aws_region}")
        oss_client = get_oss_client(CFG.opensearch.collection_id, CFG.aws_region)
    else:
        logger.info(f">>> Storing video embeddings locally..")
        all_embs = []
        all_times = []
        all_inds = []

    for batch_imgs, batch_times, batch_inds in tqdm.tqdm(video_loader):

        batch_embs = get_image_embeddings(
            image_input=batch_imgs,
            image_processor=model.processor_image,
            image_model=model.model_image,
            device=CFG.device,
            return_tensors=True)

        if CFG.smoothing.enabled:
            batch_embs = model.smoothing(batch_embs)
            batch_times = batch_times[padding:-padding]
            batch_inds = batch_inds[padding:-padding]

        if do_index:
            oss_ingest_batch(CFG.opensearch.index_name, batch_embs, batch_times, video_name, oss_client)
        else:
            all_embs.append(batch_embs)
            all_times.append(batch_times)
            all_inds.append(batch_inds)
            

    logger.info(f">>> Total processed frames: {len(video_loader)}..")
    if do_index:
        return json.dumps({'Indexed batches': len(video_loader)})
    return torch.cat(all_embs), np.concatenate(all_times), np.concatenate(all_inds)


def search_query(model, image=None, text=None, search_size=None, k=None, time_offset=None, search_name=None):
    logger.info(f'search_query: image={image}')
    logger.info(f'search_query: text={text}')
    logger.info(f'search_query: search_size={search_size}')
    logger.info(f'search_query: k={k}')
    logger.info(f'search_query: time_offset={time_offset}')
    logger.info(f'search_query: search_name={search_name}')
    # calculate embeddings
    if image == None and text == None:
        raise AttributeError("Neither image nor text was provided")

    vec = 0
    if image != None:
        image = Image.open(io.BytesIO(base64.b64decode(image)))
        vec += get_image_embeddings(
            image_input=image,
            image_processor=model.processor_image,
            image_model=model.model_image,
            return_tensors=True,
            device=CFG.device
        )

    if text != None:
        vec += get_text_embeddings(
            text_input=text,
            text_processor=model.processor_text,
            text_model=model.model_text,
            return_tensors=True,
            device=CFG.device
        )

    if image != None and text != None:
        logger.info("both text and image are provided. vec = vec / 2.0")
        vec /= 2.0

    vec = vec.tolist()[0]

    # form a query
    client = get_oss_client(CFG.opensearch.collection_id, CFG.aws_region)
    query = get_oss_query_for(vec, search_size=search_size, k=k, search_name=search_name)
    results = client.search(body=query, index=CFG.opensearch.index_name)

    # parse results
    results = parse_opensearch_results(results)
    results = {
        k: find_clusters(v, t_offset=time_offset, as_dict=True)
        for (k, v) in results.items()
    }
    return results
