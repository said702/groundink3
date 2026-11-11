import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:groundink/services/character_ink.dart';
import 'package:groundink/services/letter_mask.dart';

/// The deterministic prototypes the golden values were computed with.
Float32List prototypes() {
  final out = Float32List(32 * 64 * 64);
  for (var k = 0; k < 32; k++) {
    for (var y = 0; y < 64; y++) {
      for (var x = 0; x < 64; x++) {
        out[(k * 64 + y) * 64 + x] =
            (math.sin(0.13 * (k + 1) * x + 0.07 * (k + 2) * y + k) * 2);
      }
    }
  }
  return out;
}

void main() {
  test('letter masks match the evaluation (OpenCV) values', () {
    // Golden values: mask computation of the evaluation in Python with
    // OpenCV resizing, for wide, tall, down-scaled and cut-off boxes.
    final golden = jsonDecode(
            File('test/fixtures/letter_mask_golden.json').readAsStringSync())
        as Map<String, dynamic>;
    final protos = prototypes();
    for (final c in (golden['cases'] as List).cast<Map<String, dynamic>>()) {
      final box = (c['box'] as List).cast<int>();
      final actual = LetterMask.probabilities(
        coefficients: Float32List.fromList(
            [for (final v in c['coefficients'] as List) (v as num).toDouble()]),
        prototypes: protos,
        cx: (c['cx'] as num).toDouble(),
        cy: (c['cy'] as num).toDouble(),
        w: (c['w'] as num).toDouble(),
        h: (c['h'] as num).toDouble(),
        left: c['left'] as int,
        top: c['top'] as int,
        scaledWidth: c['nw'] as int,
        scaledHeight: c['nh'] as int,
        cropWidth: c['cropWidth'] as int,
        cropHeight: c['cropHeight'] as int,
        originX: c['originX'] as int,
        originY: c['originY'] as int,
        box: box,
      );
      final expected = (c['expected'] as List).cast<num>();
      expect(actual.length, expected.length);
      var worst = 0.0;
      for (var i = 0; i < expected.length; i++) {
        worst = math.max(worst, (actual[i] - expected[i]).abs());
      }
      expect(worst, lessThan(1e-4), reason: 'box $box');
    }
  });

  test('overlapping ink goes to the letter whose mask covers it', () {
    // Two strokes; the second letter's box also covers the first stroke.
    final image = img.Image(width: 40, height: 20);
    img.fill(image, color: img.ColorRgb8(255, 255, 255));
    img.fillRect(image,
        x1: 8, y1: 4, x2: 10, y2: 15, color: img.ColorRgb8(0, 0, 0));
    img.fillRect(image,
        x1: 20, y1: 4, x2: 22, y2: 15, color: img.ColorRgb8(0, 0, 0));
    final boxes = [
      [4, 2, 14, 18],
      [6, 2, 26, 18],
    ];
    Float32List mask(List<int> box, bool Function(int x) covers) {
      final w = box[2] - box[0], h = box[3] - box[1];
      final out = Float32List(w * h);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          out[y * w + x] = covers(box[0] + x) ? 0.9 : 0.05;
        }
      }
      return out;
    }

    final probs = [
      mask(boxes[0], (x) => x < 12),
      mask(boxes[1], (x) => x >= 16),
    ];
    final inks = CharacterInk.extractWithMasks(image, boxes, probs);
    // Without masks the second box takes the first stroke as well.
    final plain = CharacterInk.extract(image, boxes[1]);
    expect(plain.any((r) => r[0] == 8), isTrue);

    expect(inks[0], isNotEmpty);
    expect(inks[0].every((r) => r[0] == 8 && r[2] == 3), isTrue);
    expect(inks[1], isNotEmpty);
    expect(inks[1].every((r) => r[0] == 20 && r[2] == 3), isTrue);
  });

  test('ink below the mask threshold is not colored', () {
    final image = img.Image(width: 20, height: 20);
    img.fill(image, color: img.ColorRgb8(255, 255, 255));
    img.fillRect(image,
        x1: 8, y1: 4, x2: 10, y2: 15, color: img.ColorRgb8(0, 0, 0));
    final box = [4, 2, 14, 18];
    final weak = Float32List((box[2] - box[0]) * (box[3] - box[1]))
      ..fillRange(0, (box[2] - box[0]) * (box[3] - box[1]), 0.15);
    expect(CharacterInk.extractWithMasks(image, [box], [weak]).single,
        isEmpty);
  });
}
