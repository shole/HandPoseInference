using UnityEngine;

namespace MediaPipe.HandPose {

//
// Image processing part of the hand pipeline class
//

partial class HandPipeline
{
    void RunPipeline(Texture input)
    {
        var cs = _resources.compute;

        // Letterboxing scale factor
        var scale = new Vector2
          (Mathf.Max((float)input.height / input.width, 1),
           Mathf.Max(1, (float)input.width / input.height));

        // Image scaling and padding
        cs.SetInt("_spad_width", InputWidth);
        cs.SetVector("_spad_scale", scale);
        cs.SetTexture(0, "_spad_input", input);
        cs.SetBuffer(0, "_spad_output", _detector.palm.InputBuffer);
        cs.Dispatch(0, InputWidth / 8, InputWidth / 8, 1);

        // Palm detection
        _detector.palm.ProcessInput();

        // Resolve which detection feeds which hand slot (stable tracking)
        var detections = _detector.palm.Detections;
        var palmIndex = ResolveHandAssignment(detections);

        // Debug: log each detected palm with its assigned slot
        for (var i = 0; i < detections.Length; i++)
            Debug.Log($"[HandPipeline] Palm [{i}] → slot {palmIndex[i]} score={detections[i].score:F2} center={detections[i].center}");

        // Per-hand: region update → crop → landmark inference → postprocess
        cs.SetFloat("_bbox_dt", Time.deltaTime);
        cs.SetBuffer(1, "_bbox_count", _detector.palm.CountBuffer);
        cs.SetBuffer(1, "_bbox_palm", _detector.palm.DetectionBuffer);
        cs.SetBuffer(1, "_bbox_region", _buffer.region);

        cs.SetTexture(2, "_crop_input", input);
        cs.SetBuffer(2, "_crop_region", _buffer.region);
        cs.SetBuffer(2, "_crop_output", _detector.landmark.InputBuffer);

        cs.SetFloat("_post_dt", Time.deltaTime);
        cs.SetFloat("_post_scale", scale.y);
        cs.SetBuffer(3, "_post_input", _detector.landmark.OutputBuffer);
        cs.SetBuffer(3, "_post_region", _buffer.region);
        cs.SetBuffer(3, "_post_output", _buffer.filter);

        for (var hand = 0; hand < _maxHands; hand++)
        {
            cs.SetInt("_bbox_hand_index", hand);
            cs.SetInt("_bbox_palm_index", palmIndex[hand]);
            cs.Dispatch(1, 1, 1, 1);

            cs.SetInt("_crop_hand_index", hand);
            cs.Dispatch(2, CropSize / 8, CropSize / 8, 1);

            _detector.landmark.ProcessInput();

            cs.SetInt("_post_hand_index", hand);
            cs.Dispatch(3, 1, 1, 1);
        }

        UpdateTrackingState(detections, palmIndex);

        // Read cache invalidation
        InvalidateReadCache();
    }

    // Returns palmIndex[handSlot] = detection index to use for that slot.
    // If palmIndex[slot] >= detections.Length, bbox_kernel skips the slot.
    int[] ResolveHandAssignment(System.ReadOnlySpan<BlazePalm.PalmDetector.Detection> detections)
    {
        var assignment = new int[_maxHands];
        for (var i = 0; i < _maxHands; i++) assignment[i] = i; // default: slot i ← detection i

        if (!_trackInitialized || detections.Length == 0)
            return assignment;

        if (detections.Length == 1 && _maxHands >= 2)
        {
            // Assign the single detection to whichever slot it matches better
            if (MatchCost(detections[0], 1) < MatchCost(detections[0], 0))
            {
                assignment[0] = 1; // out of range → slot 0 keeps previous region
                assignment[1] = 0;
            }
            // else: default (slot 0 ← detection 0, slot 1 skips)
            return assignment;
        }

        if (detections.Length >= 2 && _maxHands >= 2)
        {
            // Try both permutations; pick the one with lower total cost
            var cost01 = MatchCost(detections[0], 0) + MatchCost(detections[1], 1);
            var cost10 = MatchCost(detections[1], 0) + MatchCost(detections[0], 1);
            if (cost10 < cost01)
            {
                assignment[0] = 1;
                assignment[1] = 0;
            }
        }

        return assignment;
    }

    float MatchCost(BlazePalm.PalmDetector.Detection det, int slot)
    {
        var distCost = Vector2.Distance(det.center, _trackCenter[slot]);
        var up = det.ring - det.wrist;
        var angle = Mathf.Atan2(up.y, up.x) - Mathf.PI / 2;
        var angleDiff = Mathf.Abs(Mathf.DeltaAngle(
            angle * Mathf.Rad2Deg,
            _trackAngle[slot] * Mathf.Rad2Deg)) * Mathf.Deg2Rad;
        return distCost + 0.3f * angleDiff;
    }

    void UpdateTrackingState(System.ReadOnlySpan<BlazePalm.PalmDetector.Detection> detections, int[] palmIndex)
    {
        for (var slot = 0; slot < _maxHands; slot++)
        {
            var di = palmIndex[slot];
            if (di < detections.Length)
            {
                var det = detections[di];
                _trackCenter[slot] = det.center;
                var up = det.ring - det.wrist;
                _trackAngle[slot] = Mathf.Atan2(up.y, up.x) - Mathf.PI / 2;
                _trackLostTime[slot] = 0f;
            }
            else
            {
                _trackLostTime[slot] += Time.deltaTime;
            }
        }
        _trackInitialized = true;
    }
}

} // namespace MediaPipe.HandPose
