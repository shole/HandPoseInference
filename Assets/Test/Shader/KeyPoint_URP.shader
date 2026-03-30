Shader "Hidden/MediaPipe/HandPose/Visualizer/KeyPoint_URP"
{
    HLSLINCLUDE

    //
    // URP version of the hand key point / skeleton visualizer material.
    // Logic is identical to KeyPoint.shader (Built-in pipeline).
    // Use this shader in projects running Universal Render Pipeline.
    //

    #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

    StructuredBuffer<float4> _KeyPoints;
    uint _HandKeyPointOffset;

    // Coloring function
    float3 DepthToColor(float z)
    {
        float3 c = lerp(1, float3(0, 0, 1), saturate(z * 2));
        return lerp(c, float3(1, 0, 0), saturate(z * -2));
    }

    //
    // Vertex shader for key points (circles)
    //
    void VertexKeys(uint vid : SV_VertexID,
                    uint iid : SV_InstanceID,
                    out float4 position : SV_Position,
                    out float4 color : COLOR)
    {
        float3 p = _KeyPoints[_HandKeyPointOffset + iid].xyz;

        uint fan = vid / 3;
        uint segment = vid % 3;

        float theta = (fan + segment - 1) * PI / 16;
        float radius = (segment > 0) * 0.08 * (max(0, -p.z) + 0.1);
        p.xy += float2(cos(theta), sin(theta)) * radius;

        position = TransformObjectToHClip(p);
        color = float4(DepthToColor(p.z), 0.8);
    }

    //
    // Vertex shader for bones (line segments)
    //
    void VertexBones(uint vid : SV_VertexID,
                     uint iid : SV_InstanceID,
                     out float4 position : SV_Position,
                     out float4 color : COLOR)
    {
        uint finger = iid / 4;
        uint segment = iid % 4;

        uint i = min(4, finger) * 4 + segment + vid;
        uint root = finger > 1 && finger < 5 ? i - 3 : 0;

        i = max(segment, vid) == 0 ? root : i;

        float3 p = _KeyPoints[_HandKeyPointOffset + i].xyz;

        position = TransformObjectToHClip(p);
        color = float4(DepthToColor(p.z), 0.8);
    }

    //
    // Common fragment shader (simple fill)
    //
    float4 Fragment(float4 position : SV_Position,
                    float4 color : COLOR) : SV_Target
    {
        return color;
    }

    ENDHLSL

    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Transparent"
        }
        ZWrite Off ZTest Always Cull Off
        Blend SrcAlpha OneMinusSrcAlpha
        Pass
        {
            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex VertexKeys
            #pragma fragment Fragment
            ENDHLSL
        }
        Pass
        {
            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex VertexBones
            #pragma fragment Fragment
            ENDHLSL
        }
    }
}
