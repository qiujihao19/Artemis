#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_spi_model, build_kmeans
from .multimodal_projector.builder import build_vision_projector

from artemis.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, BOX_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_BBOX_TOKEN
import random

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_spi_model", None):
            self.spi_model = build_spi_model(config)
        if getattr(config, 'k_means', None):
            self.k_means = build_kmeans(config)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_spi_model(self):
        spi_model = getattr(self, 'spi_model', None)
        if type(spi_model) is list:
            spi_model = spi_model[0]
        return spi_model
    
    def get_kmeans_model(self):
        k_means = getattr(self, 'k_means', None)
        if type(k_means) is list:
            k_means = k_means[0]
        return k_means   
    def initialize_vision_modules(self, model_args, fsdp=None):
        # ==============================================
        image_tower = model_args.image_tower
        spi_model = model_args.spi_model
        k_means = model_args.k_means
        assert image_tower is not None
        # ==============================================
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # ==========================================================================

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()

        self.config.mm_spi_model = spi_model 
        self.config.k_means = k_means       
        if spi_model:
            if self.get_spi_model() is None:
                spi_model = build_spi_model(model_args)
                self.spi_model = spi_model
        if k_means:
            if self.get_kmeans_model() is None:
                k_means = build_kmeans(model_args)
                self.k_means = k_means
                


        # ==========================================================================

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # ==========================================================================
        if image_tower is not None :  # TODO: support different hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        else:
            self.config.mm_hidden_size = getattr(image_tower, 'hidden_size', -1)
        # ===================================================================================

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_spi_model(self):
        return self.get_model().get_spi_model()
    
    def get_kmeans_model(self):
        return self.get_model().get_kmeans_model()
    
    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def endcode_mlvl_feature(self, images4box):
        mlvl_spi_features = self.get_model().get_image_tower().forward_spi(images4box)
        return mlvl_spi_features
    
    def encode_region_feature(self, mlvl_spi_features, bboxes):    
        mlvl_spi_features = self.get_model().get_spi_model()(mlvl_spi_features, bboxes)
        return mlvl_spi_features

    def encode_videos(self, videos):  # [mini_b, n, c]
        b, N, C = videos.shape
        video_features = self.get_model().mm_projector(videos)
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, images4box, bboxes, offset
    ):
        # ====================================================================================================
        image_tower = self.get_image_tower()
        spi_model = self.get_spi_model()
        k_means = self.get_kmeans_model()
        if (image_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        '''
            images is a list, if batch_size=6
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(nv, c),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(nv, c),   # sample 5
                video(nv, c),   # sample 5
                video(nv, c),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
            will be converted to image_features, all video_feature will be flatten as image
            [
                [n, c],                  # sample 1
                [n, c),                  # sample 2
                [nv, c],       # sample 3
                [n, c],                  # sample 4
                [n, c],                  # sample 4
                [nv, c],       # sample 5
                [nv, c],       # sample 5
                [nv, c],       # sample 6
                [n, c],                  # sample 6
            ]
        '''
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]

        video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 2]
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # mini_b c h w
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []  # mini_b nv c

        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4:  # batch consists of images, [mini_b, c, h, w]
            if image_tower is not None:
                image_features_minibatch = self.encode_images(images_minibatch)  # [mini_b, l, c]
            else:
                image_features_minibatch = torch.randn(1).to(self.device)  # dummy feature for video-only training under tuning
            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        if getattr(videos_minibatch, 'ndim', 0) == 3:  # batch consists of videos, [mini_b, nv, c]
            video_features_minibatch = self.encode_videos(videos_minibatch)  # fake list [mini_b, nv, c]
            for i, pos in enumerate(video_idx):
                tmp_image_features[pos] = video_features_minibatch[i]
      
        if spi_model and bboxes is not None and sum([len(bbox) for bbox in bboxes]) > 0 and len(images4box) == len(bboxes):
            images4box = torch.stack(images4box) if type(images4box) is list else images4box
            mlvl_spi_features = self.endcode_mlvl_feature(images4box)
            mlvl_spi_features = self.encode_region_feature(mlvl_spi_features, bboxes)
        else:
            mlvl_spi_features = [None for _ in range(len(input_ids))]
        
        if mlvl_spi_features[0] is not None and k_means and len(bboxes) - offset >= k_means.n_clusters:
            batch_size = input_ids.shape[0]
            assert len(mlvl_spi_features) % batch_size==0
            num_trackbox = len(mlvl_spi_features) // batch_size
            mlvl_spi_features_cat = torch.cat(mlvl_spi_features, dim=0)#tensor(bs*num_trackbox,4096)
            mlvl_spi_features_split = list(torch.split(mlvl_spi_features_cat, num_trackbox, dim=0))#[tensor(num_trackbox,4096)],len(list)=bs
            input_bbox_feature = []
            for bbox_feature in mlvl_spi_features_split:
                kmeans_box_feature = bbox_feature[1+offset:,:].unsqueeze(dim=0)
                result = self.get_model().get_kmeans_model()(kmeans_box_feature)
                labels_ = result.labels
                labels_ids = torch.unique(labels_)
                # box_idx = [random.choice((labels == i).nonzero())[1] for i in range(self.kmeans_box)]
                box_idx = [random.choice((labels_ == i.item()).nonzero())[1] for i in labels_ids]
                if len(box_idx) < k_means.n_clusters:
                    for _ in range(k_means.n_clusters - len(box_idx)):
                        box_idx.append(box_idx[-1])
                box_idx = sorted(box_idx)
                box_idx = torch.stack(box_idx, dim=0)
                kmeans_box_feature = kmeans_box_feature.squeeze(dim=0)
                remain_box_feature = bbox_feature[:1+offset]
                remain_box_feature = remain_box_feature.unsqueeze(dim=0) if len(remain_box_feature.shape) == 1 else remain_box_feature
                input_bbox_feature.append(torch.cat([remain_box_feature, kmeans_box_feature[box_idx]], dim=0))
                
            mlvl_spi_features = input_bbox_feature
        elif mlvl_spi_features[0] is not None:        
            batch_size = input_ids.shape[0]
            assert len(mlvl_spi_features) % batch_size==0
            num_trackbox = len(mlvl_spi_features) // batch_size
            mlvl_spi_features_cat = torch.cat(mlvl_spi_features, dim=0)#tensor(bs*num_trackbox,4096)
            mlvl_spi_features = list(torch.split(mlvl_spi_features_cat, num_trackbox, dim=0))               

        new_tmp = []
        for image in tmp_image_features:
            # print(len(new_tmp), len(image))
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
                # print('add video')
            else:
                new_tmp.append(image)
        image_features = new_tmp
        # print(len(image_features), *[i.shape for i in image_features])
        # print(len(image_features), image_features[0].shape)
        # ====================================================================================================

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_ids_feature in enumerate(zip(input_ids, mlvl_spi_features)):
            cur_input_ids = cur_ids_feature[0]
            spi_feat = cur_ids_feature[1]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_boxes = (cur_input_ids == BOX_TOKEN_INDEX).sum()
            # print(num_images, cur_input_ids)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            
            med_cur_labels_noim = []
            med_cur_input_embeds = []
            for slice_cur_input_ids_noim, slice_cur_labels_noim in zip(cur_input_ids_noim, cur_labels_noim):
                if BOX_TOKEN_INDEX in slice_cur_input_ids_noim:
                    bbox_token_indices = [-1] + torch.where(slice_cur_input_ids_noim == BOX_TOKEN_INDEX)[0].tolist() + [slice_cur_input_ids_noim.shape[0]]
                    box_cur_input_ids_noim = []
                    box_cur_labels = slice_cur_labels_noim
                    box_cur_labels_noim = []
                    for i in range(len(bbox_token_indices) - 1):
                        box_cur_input_ids_noim.append(slice_cur_input_ids_noim[bbox_token_indices[i]+1:bbox_token_indices[i+1]])
                        box_cur_labels_noim.append(box_cur_labels[bbox_token_indices[i]+1:bbox_token_indices[i+1]]) 
                    box_split_sizes = [x.shape[0] for x in box_cur_labels_noim]
                    box_cur_input_embeds = self.get_model().embed_tokens(torch.cat(box_cur_input_ids_noim))   
                                    
                    box_cur_input_embeds_no_im = torch.split(box_cur_input_embeds, box_split_sizes, dim=0)
                    box_cur_new_input_embeds = []
                    box_cur_new_labels = []
                    cur_box_idx = 0
                    for i in range(num_boxes + 1):
                        box_cur_new_input_embeds.append(box_cur_input_embeds_no_im[i])
                        box_cur_new_labels.append(box_cur_labels_noim[i])
                        if i < num_boxes:
                            # print(cur_image_idx)
                            cur_bbox_features = spi_feat[cur_box_idx].unsqueeze(0)
                            cur_box_idx += 1
                            box_cur_new_input_embeds.append(cur_bbox_features)
                            box_cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    med_cur_input_embeds.append(torch.cat(box_cur_new_input_embeds))
                    med_cur_labels_noim.append(torch.cat(box_cur_new_labels))
                else:
                    med_cur_input_embeds.append(self.get_model().embed_tokens(slice_cur_input_ids_noim))
                    med_cur_labels_noim.append(slice_cur_labels_noim)
 
            cur_input_embeds_no_im = med_cur_input_embeds
            cur_labels_noim = med_cur_labels_noim
 

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # print(cur_image_idx)
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_bbox_token:
            tokenizer.add_tokens([DEFAULT_BBOX_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            image_config = self.get_image_tower().config
            image_config.bbox_token = tokenizer.convert_tokens_to_ids([DEFAULT_BBOX_TOKEN])[0]
            
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
