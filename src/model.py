# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================
#
# CreativeML Open RAIL-M License
#
# Section I: PREAMBLE

# Multimodal generative models are being widely adopted and used, and have the potential to transform the way artists, among other individuals, conceive and benefit from AI or ML technologies as a tool for content creation.

# Notwithstanding the current and potential benefits that these artifacts can bring to society at large, there are also concerns about potential misuses of them, either due to their technical limitations or ethical considerations.

# In short, this license strives for both the open and responsible downstream use of the accompanying model. When it comes to the open character, we took inspiration from open source permissive licenses regarding the grant of IP rights. Referring to the downstream responsible use, we added use-based restrictions not permitting the use of the Model in very specific scenarios, in order for the licensor to be able to enforce the license in case potential misuses of the Model may occur. At the same time, we strive to promote open and responsible research on generative models for art and content generation.

# Even though downstream derivative versions of the model could be released under different licensing terms, the latter will always have to include - at minimum - the same use-based restrictions as the ones in the original license (this license). We believe in the intersection between open and responsible AI development; thus, this License aims to strike a balance between both in order to enable responsible open-science in the field of AI.

# This License governs the use of the model (and its derivatives) and is informed by the model card associated with the model.

# NOW THEREFORE, You and Licensor agree as follows:

# 1. Definitions

# - "License" means the terms and conditions for use, reproduction, and Distribution as defined in this document.
# - "Data" means a collection of information and/or content extracted from the dataset used with the Model, including to train, pretrain, or otherwise evaluate the Model. The Data is not licensed under this License.
# - "Output" means the results of operating a Model as embodied in informational content resulting therefrom.
# - "Model" means any accompanying machine-learning based assemblies (including checkpoints), consisting of learnt weights, parameters (including optimizer states), corresponding to the model architecture as embodied in the Complementary Material, that have been trained or tuned, in whole or in part on the Data, using the Complementary Material.
# - "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.
# - "Complementary Material" means the accompanying source code and scripts used to define, run, load, benchmark or evaluate the Model, and used to prepare data for training or evaluation, if any. This includes any accompanying documentation, tutorials, examples, etc, if any.
# - "Distribution" means any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access.
# - "Licensor" means the copyright owner or entity authorized by the copyright owner that is granting the License, including the persons or entities that may have rights in the Model and/or distributing the Model.
# - "You" (or "Your") means an individual or Legal Entity exercising permissions granted by this License and/or making use of the Model for whichever purpose and in any field of use, including usage of the Model in an end-use application - e.g. chatbot, translator, image generator.
# - "Third Parties" means individuals or legal entities that are not under common control with Licensor or You.
# - "Contribution" means any work of authorship, including the original version of the Model and any modifications or additions to that Model or Derivatives of the Model thereof, that is intentionally submitted to Licensor for inclusion in the Model by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Model, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
# - "Contributor" means Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Model.

# Section II: INTELLECTUAL PROPERTY RIGHTS

# Both copyright and patent grants apply to the Model, Derivatives of the Model and Complementary Material. The Model and Derivatives of the Model are subject to additional terms as described in Section III.

# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute the Complementary Material, the Model, and Derivatives of the Model.
# 3. Grant of Patent License. Subject to the terms and conditions of this License and where and as applicable, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this paragraph) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Model and the Complementary Material, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Model to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model and/or Complementary Material or a Contribution incorporated within the Model and/or Complementary Material constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for the Model and/or Work shall terminate as of the date such litigation is asserted or filed.

# Section III: CONDITIONS OF USAGE, DISTRIBUTION AND REDISTRIBUTION

# 4. Distribution and Redistribution. You may host for Third Party remote access purposes (e.g. software-as-a-service), reproduce and distribute copies of the Model or Derivatives of the Model thereof in any medium, with or without modifications, provided that You meet the following conditions:
# Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5. This provision does not apply to the use of Complementary Material.
# You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License;
# You must cause any modified files to carry prominent notices stating that You changed the files;
# You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model, Derivatives of the Model.
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions - respecting paragraph 4.a. - for use, reproduction, or Distribution of Your modifications, or for any such Derivatives of the Model as a whole, provided Your use, reproduction, and Distribution of the Model otherwise complies with the conditions stated in this License.
# 5. Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
# 6. The Output You Generate. Except as set forth herein, Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses. No use of the output can contravene any provision as stated in the License.

# Section IV: OTHER PROVISIONS

# 7. Updates and Runtime Restrictions. To the maximum extent permitted by law, Licensor reserves the right to restrict (remotely or otherwise) usage of the Model in violation of this License, update the Model through electronic means, or modify the Output of the Model based on updates. You shall undertake reasonable efforts to use the latest version of the Model.
# 8. Trademarks and related. Nothing in this License permits You to make use of Licensors’ trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between the parties; and any rights not expressly granted herein are reserved by the Licensors.
# 9. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Model and the Complementary Material (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Model, Derivatives of the Model, and the Complementary Material and assume any risks associated with Your exercise of permissions under this License.
# 10. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Model and the Complementary Material (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
# 11. Accepting Warranty or Additional Liability. While redistributing the Model, Derivatives of the Model and the Complementary Material thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
# 12. If any provision of this License is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.

# END OF TERMS AND CONDITIONS




# Attachment A

# Use Restrictions

# You agree not to use the Model or Derivatives of the Model:
# - In any way that violates any applicable national, federal, state, local or international law or regulation;
# - For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# - To generate or disseminate verifiably false information and/or content with the purpose of harming others;
# - To generate or disseminate personal identifiable information that can be used to harm an individual;
# - To defame, disparage or otherwise harass others;
# - For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
# - For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
# - To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# - For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
# - To provide medical advice and medical results interpretation;
# - To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).

import torch
from einops import rearrange, repeat
from torch import nn, einsum

import torch.nn.functional as F
from ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion
from ldm.util import default
from ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlock
from ldm.modules.attention import CrossAttention as CrossAttention
from ldm.util import log_txt_as_img, exists, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torchvision.utils import make_grid
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
import numpy as np
from typing import Optional, List, Type, Set


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, old_weight=None, old_bias=None, device=None, dtype=None, r=4
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.old_weight = old_weight
        self.old_bias = old_bias
        self.lora_down = torch.nn.Parameter(torch.zeros((out_features, r), **factory_kwargs))
        self.lora_up = torch.nn.Parameter(torch.zeros((r, in_features), **factory_kwargs))

    def forward(self, x):
        weight = self.old_weight + self.lora_down @ self.lora_up
        output = F.linear(x, weight, bias=self.old_bias)
        return output

    def realize_as_lorsa(self):
        return self.lora_up, self.lora_down

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


class LorsaInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, old_weight=None, old_bias=None, device=None, dtype=None, r=4, shrinkage_threshold=0.001,
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.old_weight = old_weight
        self.old_bias = old_bias
        self.lora_down = torch.nn.Parameter(torch.zeros((out_features, r), **factory_kwargs))
        self.lora_up = torch.nn.Parameter(torch.zeros((r, in_features), **factory_kwargs))
        self.sparsity = torch.nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs))
        self.shrinkage_threshold = shrinkage_threshold

    def forward(self, x):
        weight = self.old_weight + self.sparsity + self.lora_down @ self.lora_up
        output = F.linear(x, weight, bias=self.old_bias)
        return output

    def realize_as_lorsa(self):
        return self.lora_up, self.lora_down, self.sparsity

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)

# Possible bug causes
# missing dropout in InjectedLinear
# existence of InjectedLinear
# not doing the same masking as their change_forward
# not commenting out their change_forward
# something seems fucked up about my initialization because it can't even generate people
# would be faster if my injected layers add weights and biases rather than running multiple times, will need to get rid of dropout


class InjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, old_weight=None, old_bias=None, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.injected_linear_weight = torch.nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs))
        self.injected_linear_bias = torch.nn.Parameter(torch.zeros((out_features), **factory_kwargs)) if bias else None
        self.old_weight = old_weight
        self.old_bias = old_bias
        # self.selector = nn.Identity()

    def forward(self, x):
        weight = self.old_weight + self.injected_linear_weight
        bias = self.old_bias + self.injected_linear_bias if self.old_bias is not None else None
        output = F.linear(x, weight, bias=bias)
        return output

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.linear.weight.device
        ).to(self.linear.weight.dtype)


# Modified from https://github.com/cloneofsimo/lora/blob/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd/lora_diffusion/lora.py#L189
def _find_modules(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] =
    [
        LoraInjectedLinear,
        LorsaInjectedLinear,
        InjectedLinear,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                # print(f'considering {fullname} of type {module}')
                while path:
                    parent = parent.get_submodule(path.pop(0))
                    # print(f'parent class is {parent}, condition is {any([isinstance(parent, _class) for _class in exclude_children_of])}')
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    # print(f'skipping parent class {parent}')
                    continue
                # Skip this linear if it's one of our special ones
                if any([keyword in fullname for keyword in ['injected', 'sparse', 'lora', 'extra']]):
                    # print(f'skipping {fullname}')
                    continue
                # Sanity check: only edit crossattn k and v weights
                # if not any([keyword in fullname for keyword in ['to_k', 'to_v']]):
                #     # print(f'skipping {fullname}')
                #     continue
                # Otherwise, yield it
                # print(f'keeping {fullname}')
                yield parent, name, module

class CustomDiffusion(LatentDiffusion):
    def __init__(self,
                 freeze_model='crossattn-kv',
                 lora_rank=None,
                 shrinkage_threshold=0.0,
                 cond_stage_trainable=False,
                 add_token=False,
                 *args, **kwargs):

        self.freeze_model = freeze_model
        self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        self.shrinkage_threshold = shrinkage_threshold
        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)

        self.require_grad_params = None

        def change_checkpoint(model):
            for layer in model.children():
                if type(layer) == BasicTransformerBlock:
                    layer.checkpoint = False
                else:
                    change_checkpoint(layer)

        change_checkpoint(self.model.diffusion_model)

        # def new_forward(self, x, context=None, mask=None):
        #     h = self.heads
        #     crossattn = False
        #     if context is not None:
        #         crossattn = True
        #     q = self.to_q(x)
        #     context = default(context, x)
        #     k = self.to_k(context)
        #     v = self.to_v(context)

        #     if crossattn:
        #         modifier = torch.ones_like(k)
        #         modifier[:, :1, :] = modifier[:, :1, :]*0.
        #         k = modifier*k + (1-modifier)*k.detach()
        #         v = modifier*v + (1-modifier)*v.detach()

        #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        #     sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        #     attn = sim.softmax(dim=-1)

        #     out = einsum('b i j, b j d -> b i d', attn, v)
        #     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        #     return self.to_out(out)

        # def change_forward(model):
        #     for name, layer in model.named_children():
        #         if type(layer) == CrossAttention and 'attn2' in name:
        #             bound_method = new_forward.__get__(layer, layer.__class__)
        #             setattr(layer, 'forward', bound_method)
        #         else:
        #             change_forward(layer)

        # change_forward(self.model.diffusion_model)

        # # At first, set all parameters to not be updated
        for x in self.model.diffusion_model.named_parameters():
            x[1].requires_grad = False

        # if self.freeze_model == 'crossattn-kv':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' not in x[0]:
        #             x[1].requires_grad = False
        #         elif not ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0]):
        #             x[1].requires_grad = False
        #         else:
        #             x[1].requires_grad = True
        # elif self.freeze_model == 'crossattn-kv-lora':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' not in x[0]:
        #             x[1].requires_grad = False
        #         elif not ('lora_up' in x[0] or 'lora_down' in x[0]):
        #             x[1].requires_grad = False
        #         else:
        #             x[1].requires_grad = True
        # elif self.freeze_model == 'crossattn-kv-lorsa':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' not in x[0]:
        #             x[1].requires_grad = False
        #         elif not ('lora_up' in x[0] or 'lora_down' in x[0] or 'sparse_linear' in x[0]):
        #             x[1].requires_grad = False
        #         else:
        #             x[1].requires_grad = True
        # elif self.freeze_model == 'crossattn':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' not in x[0]:
        #             x[1].requires_grad = False
        #         elif not 'attn2' in x[0]:
        #             x[1].requires_grad = False
        #         else:
        #             x[1].requires_grad = True
        # elif self.freeze_model == 'weight':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'weight' in x[0]:
        #             x[1].requires_grad = True
        #         else:
        #             x[1].requires_grad = False
        # elif self.freeze_model == 'weight-lora':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'weight' not in x[0]:
        #             x[1].requires_grad = False
        #         elif not ('lora_up' in x[0] or 'lora_down' in x[0]):
        #             x[1].requires_grad = False
        #         else:
        #             x[1].requires_grad = True
        # elif self.freeze_model == 'weight-lorsa':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'weight' not in x[0]:
        #             x[1].requires_grad = False
        #         elif not ('lora_up' in x[0] or 'lora_down' in x[0] or 'sparse_linear' in x[0]):
        #             x[1].requires_grad = False
        #         else:
        #             x[1].requires_grad = True
        #breakpoint()

    # Modified from https://github.com/cloneofsimo/lora/blob/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd/lora_diffusion/lora.py#L254
    def inject_trainable_lora(self, lora_rank):
        def change_forward(model):
            self.require_grad_params = []
            self.names = []

            for _module, name, _child_module in _find_modules(
                model, None, search_class=[nn.Linear]
            ):
                _tmp = LoraInjectedLinear(
                    _child_module.in_features,
                    _child_module.out_features,
                    _child_module.bias is not None,
                    _child_module.weight,
                    _child_module.bias,
                    device=_child_module.weight.device,
                    dtype=_child_module.weight.dtype,
                    r=lora_rank,
                    )

                _module._modules[name] = _tmp

                # Allow the LoRSA weights to update
                self.require_grad_params.append({'params': _module._modules[name].lora_up})
                self.require_grad_params.append({'params': _module._modules[name].lora_down})
                _module._modules[name].lora_up.requires_grad = True
                _module._modules[name].lora_down.requires_grad = True
                self.names.append(name)
        change_forward(self.model.diffusion_model)


    def inject_trainable_lorsa(self, lora_rank, shrinkage_threshold):
        def change_forward(model):
            self.require_grad_params = []
            self.names = []

            for _module, name, _child_module in _find_modules(
                model, None, search_class=[nn.Linear]
            ):
                _tmp = LorsaInjectedLinear(
                    _child_module.in_features,
                    _child_module.out_features,
                    _child_module.bias is not None,
                    _child_module.weight,
                    _child_module.bias,
                    device=_child_module.weight.device,
                    dtype=_child_module.weight.dtype,
                    r=lora_rank,
                    shrinkage_threshold=shrinkage_threshold,
                    )

                _module._modules[name] = _tmp

                # Allow the LoRSA weights to update
                self.require_grad_params.append({'params': _module._modules[name].lora_up})
                self.require_grad_params.append({'params': _module._modules[name].lora_down})
                self.require_grad_params.append({'params': _module._modules[name].sparsity})
                _module._modules[name].lora_up.requires_grad = True
                _module._modules[name].lora_down.requires_grad = True
                _module._modules[name].sparsity.requires_grad = True
                self.names.append(name)
        change_forward(self.model.diffusion_model)


    def inject_trainable_linear(self):
        def change_forward(model):
            self.require_grad_params = []
            self.names = []

            for _module, name, _child_module in list(_find_modules(
                model, None, search_class=[nn.Linear]
            )):
                # print("Linear Injection : injecting linear into ", name)
                # print(type(_child_module))
                _tmp = InjectedLinear(
                    _child_module.in_features,
                    _child_module.out_features,
                    _child_module.bias is not None,
                    _child_module.weight,
                    _child_module.bias,
                    device=_child_module.weight.device,
                    dtype=_child_module.weight.dtype,
                    )
                assert torch.sum((_tmp.forward(torch.ones((1, _child_module.in_features))) - _child_module.forward(torch.ones((1, _child_module.in_features))))**2) == 0

                _module._modules[name] = _tmp

                # Allow the linear weights to update
                self.require_grad_params.append({'params': _tmp.injected_linear_weight})
                _tmp.injected_linear_weight.requires_grad = True
                if _tmp.injected_linear_bias is not None:
                    self.require_grad_params.append({'params': _tmp.injected_linear_bias})
                    _tmp.injected_linear_bias.requires_grad = True
                self.names.append(name)
        change_forward(self.model.diffusion_model)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.require_grad_params
        if params is None:
            params = list(self.model.parameters())

        # params = []
        # if self.freeze_model == 'crossattn-kv':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' in x[0]:
        #             if 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]:
        #                 params += [x[1]]
        #                 # print(x[0])
        # elif self.freeze_model == 'crossattn-kv-lora' or self.freeze_model == 'crossattn-kv-lorsa':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' in x[0]:
        #             if 'lora_up' in x[0] or 'lora_down' in x[0] or 'sparse_linear' in x[0]:
        #                 params += [x[1]]
        #                 # print(x[0])
        # elif self.freeze_model == 'crossattn':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'transformer_blocks' in x[0]:
        #             if 'attn2' in x[0]:
        #                 params += [x[1]]
        #                 # print(x[0])
        # elif self.freeze_model == 'weight-lora' or self.freeze_model == 'weight-lorsa':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'weight' in x[0]:
        #             if 'lora_up' in x[0] or 'lora_down' in x[0] or 'sparse_linear' in x[0]:
        #                 params += [x[1]]
        #                 # print(x[0])
        # elif self.freeze_model == 'weight':
        #     for x in self.model.diffusion_model.named_parameters():
        #         if 'weight' in x[0]:
        #             params += [x[1]]
        #             # print(x[0])
        # else:
        #     params = list(self.model.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.add_token:
                params = params + [{'params': self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters()}]
            else:
                params = params + [{'params': self.cond_stage_model.parameters()}]
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def p_losses(self, x_start, cond, t, mask=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_simple = (loss_simple*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_simple = loss_simple.mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = (self.logvar.to(self.device))[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_vlb = (loss_vlb*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_vlb = loss_vlb.mean([1, 2, 3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def get_input_withmask(self, batch, **args):
        out = super().get_input(batch, self.first_stage_key, **args)
        mask = batch["mask"]
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = rearrange(mask, 'b h w c -> b c h w')
        mask = mask.to(memory_format=torch.contiguous_format).float()
        out += [mask]
        return out

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            train_batch = batch[0]
            train2_batch = batch[1]
            loss_train, loss_dict = self.shared_step(train_batch)
            loss_train2, _ = self.shared_step(train2_batch)
            loss = loss_train + loss_train2
        else:
            train_batch = batch
            loss, loss_dict = self.shared_step(train_batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input_withmask(batch, **kwargs)
        loss = self(x, c, mask=mask)
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                unconditional_guidance_scale=6.
                unconditional_conditioning = self.get_learned_conditioning(len(c) * [""])
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                        unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples_scaled"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
