using MediaPipe.BlazePalm;
using MediaPipe.HandLandmark;
using UnityEngine;
using UnityEngine.Rendering;

namespace MediaPipe.HandPose {

//
// Basic implementation of the hand pipeline class
//

sealed partial class HandPipeline : System.IDisposable
{
    #region Private objects

    const int CropSize = HandLandmarkDetector.ImageSize;
    int InputWidth => _detector.palm.ImageSize;

    ResourceSet _resources;
    int _maxHands;
    (PalmDetector palm, HandLandmarkDetector landmark) _detector;
    (ComputeBuffer region, ComputeBuffer filter) _buffer;
    GlobalKeyword _keywordNchw;

    // Per-slot tracking state for detection-to-hand assignment
    UnityEngine.Vector2[] _trackCenter;
    float[] _trackAngle;
    float[] _trackLostTime;
    bool _trackInitialized;

    #endregion

    #region Object allocation/deallocation

    void AllocateObjects(ResourceSet resources, int maxHands)
    {
        _resources = resources;
        _maxHands = maxHands;

        _detector = (new PalmDetector(_resources.blazePalm),
                     new HandLandmarkDetector(_resources.handLandmark));

        var regionStructSize = sizeof(float) * 24;
        var filterBufferLength = HandLandmarkDetector.VertexCount * 2 * _maxHands;

        _buffer = (new ComputeBuffer(_maxHands, regionStructSize),
                   new ComputeBuffer(filterBufferLength, sizeof(float) * 4));

        _keywordNchw = GlobalKeyword.Create("NCHW_INPUT");
        Shader.SetKeyword(_keywordNchw, _detector.palm.InputIsNCHW);

        InitReadCache();

        _trackCenter = new UnityEngine.Vector2[_maxHands];
        _trackAngle = new float[_maxHands];
        _trackLostTime = new float[_maxHands];
        for (var i = 0; i < _maxHands; i++) _trackLostTime[i] = float.PositiveInfinity;
        _trackInitialized = false;
    }

    void DeallocateObjects()
    {
        _detector.palm.Dispose();
        _detector.landmark.Dispose();
        _buffer.region.Dispose();
        _buffer.filter.Dispose();
    }

    #endregion
}

} // namespace MediaPipe.HandPose
