using UnityEngine;

namespace MediaPipe.HandPose {

//
// Public part of the hand pipeline class
//

partial class HandPipeline
{
    #region Detection data accessors

    public const int KeyPointCount = 21;

    public enum KeyPoint
    {
        Wrist,
        Thumb1,  Thumb2,  Thumb3,  Thumb4,
        Index1,  Index2,  Index3,  Index4,
        Middle1, Middle2, Middle3, Middle4,
        Ring1,   Ring2,   Ring3,   Ring4,
        Pinky1,  Pinky2,  Pinky3,  Pinky4
    }

    public Vector3 GetKeyPoint(KeyPoint point, int handIndex = 0)
      => ReadCache[handIndex * KeyPointCount * 2 + (int)point];

    public Vector3 GetKeyPoint(int index, int handIndex = 0)
      => ReadCache[handIndex * KeyPointCount * 2 + index];

    #endregion

    #region GPU-side resource accessors

    public ComputeBuffer KeyPointBuffer
      => _buffer.filter;

    public ComputeBuffer HandRegionBuffer
      => _buffer.region;

    public ComputeBuffer HandRegionCropBuffer
      => _detector.landmark.InputBuffer;

    #endregion

    #region Public properties and methods

    public bool UseAsyncReadback { get; set; } = true;

    public int MaxHands => _maxHands;

    public HandPipeline(ResourceSet resources, int maxHands = 2)
      => AllocateObjects(resources, maxHands);

    public void Dispose()
      => DeallocateObjects();

    public void ProcessImage(Texture image)
      => RunPipeline(image);

    #endregion
}

} // namespace MediaPipe.HandPose
