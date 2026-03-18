/*
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_fragment_shader_barycentric : enable

#include "shaderio.h"
#include "nvshaders/sky_functions.h.slang"

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

///////////////////////////////////////////////////

layout(location = 0, index = 0) out vec4 out_Color;

///////////////////////////////////////////////////

void main()
{
  vec2 screenPos = ((vec2(gl_FragCoord.xy) / view.viewportf) * 2.0) - 1.0;
  
  vec4 transformed = view.skyProjMatrixI * vec4(screenPos, 1.0,  1);
  vec3 rayDir      = normalize(transformed.xyz);
  
  vec3 skyColor = evalSimpleSky(view.skyParams, rayDir);

  out_Color = vec4(skyColor, 1);
}