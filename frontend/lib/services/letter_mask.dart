import 'dart:math' as math;
import 'dart:typed_data';

/// The outline of one letter as predicted by the segmentation head of the
/// character model, as a probability per pixel of the source image.
///
/// The model returns 32 mask prototypes of 64 x 64 for the whole 256 x 256
/// input and 32 coefficients per detection. As in Ultralytics, a letter's
/// mask is `sigmoid(coefficients x prototypes)`, scaled up to 256 x 256
/// (bilinear), set to 0 outside the letter's box, and then mapped from the
/// letterboxed input back onto the word crop (bilinear again).
class LetterMask {
  static const protoCount = 32;
  static const protoSize = 64;
  static const inputSize = 256;

  /// Mask probabilities for the source pixels of [box] (`[x1, y1, x2, y2]`,
  /// row by row, `(x2 - x1) * (y2 - y1)` values).
  ///
  /// [coefficients]: the detection's 32 mask coefficients. [prototypes]:
  /// the model's `[32, 64, 64]` prototypes. [cx], [cy], [w], [h]: the
  /// detection's box in input (letterbox) coordinates. The word crop of
  /// [cropWidth] x [cropHeight] pixels starts at [originX], [originY] in the
  /// source image and fills [scaledWidth] x [scaledHeight] input pixels from
  /// [left], [top].
  static Float32List probabilities({
    required Float32List coefficients,
    required Float32List prototypes,
    required double cx,
    required double cy,
    required double w,
    required double h,
    required int left,
    required int top,
    required int scaledWidth,
    required int scaledHeight,
    required int cropWidth,
    required int cropHeight,
    required int originX,
    required int originY,
    required List<int> box,
  }) {
    const n = protoSize * protoSize;
    final small = Float64List(n);
    for (var k = 0; k < protoCount; k++) {
      final c = coefficients[k];
      if (c == 0) continue;
      final offset = k * n;
      for (var p = 0; p < n; p++) {
        small[p] += c * prototypes[offset + p];
      }
    }
    for (var p = 0; p < n; p++) {
      small[p] = 1 / (1 + math.exp(-small[p]));
    }

    // The letter's box in input pixels; the mask is 0 outside it.
    final bx1 = math.max(0, (cx - w / 2).floor());
    final by1 = math.max(0, (cy - h / 2).floor());
    final bx2 = math.min(inputSize, (cx + w / 2).ceil());
    final by2 = math.min(inputSize, (cy + h / 2).ceil());
    final bw = math.max(0, bx2 - bx1), bh = math.max(0, by2 - by1);

    // 64 -> 256, only inside the box.
    final large = Float64List(bw * bh);
    for (var v = 0; v < bh; v++) {
      final (y0, y1, fy) = _source(by1 + v, protoSize / inputSize, protoSize);
      for (var u = 0; u < bw; u++) {
        final (x0, x1, fx) =
            _source(bx1 + u, protoSize / inputSize, protoSize);
        final top0 = small[y0 * protoSize + x0] * (1 - fx) +
            small[y0 * protoSize + x1] * fx;
        final bottom0 = small[y1 * protoSize + x0] * (1 - fx) +
            small[y1 * protoSize + x1] * fx;
        large[v * bw + u] = top0 * (1 - fy) + bottom0 * fy;
      }
    }
    double inputValue(int x, int y) {
      if (x < bx1 || x >= bx2 || y < by1 || y >= by2) return 0;
      return large[(y - by1) * bw + (x - bx1)];
    }

    // Letterboxed input -> word crop -> source pixels of the letter's box.
    final outW = math.max(0, box[2] - box[0]);
    final outH = math.max(0, box[3] - box[1]);
    final out = Float32List(outW * outH);
    for (var oy = 0; oy < outH; oy++) {
      final py = box[1] + oy - originY;
      if (py < 0 || py >= cropHeight) continue;
      final (v0, v1, fy) =
          _source(py, scaledHeight / cropHeight, scaledHeight);
      for (var ox = 0; ox < outW; ox++) {
        final px = box[0] + ox - originX;
        if (px < 0 || px >= cropWidth) continue;
        final (u0, u1, fx) = _source(px, scaledWidth / cropWidth, scaledWidth);
        final a = inputValue(left + u0, top + v0) * (1 - fx) +
            inputValue(left + u1, top + v0) * fx;
        final b = inputValue(left + u0, top + v1) * (1 - fx) +
            inputValue(left + u1, top + v1) * fx;
        out[oy * outW + ox] = a * (1 - fy) + b * fy;
      }
    }
    return out;
  }

  /// Bilinear source position of output pixel [d] when resizing by
  /// [scale] (source size / output size) from [sourceSize] pixels, with the
  /// edge pixels repeated (as OpenCV's `INTER_LINEAR`).
  static (int, int, double) _source(int d, double scale, int sourceSize) {
    var s = (d + 0.5) * scale - 0.5;
    if (s < 0) s = 0;
    if (s > sourceSize - 1) s = (sourceSize - 1).toDouble();
    final s0 = s.floor();
    final s1 = math.min(s0 + 1, sourceSize - 1);
    return (s0, s1, s - s0);
  }
}
