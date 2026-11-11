import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'image_preprocessor.dart';

class CharacterInk {
  /// Minimum mask probability for ink to belong to a letter
  /// (see [extractWithMasks]).
  static const maskThreshold = 0.2;

  /// Horizontal foreground runs [x, y, length] in original image coordinates.
  static List<List<int>> extract(img.Image image, List<int> bbox) =>
      _runs(image, bbox, null);

  /// The ink of every letter of a word, using the letter outlines the
  /// character model predicts.
  ///
  /// A pixel is ink of letter `i` when it is dark inside the letter's box
  /// (the same threshold as [extract]), the letter's mask probability there
  /// is above [threshold], and no other letter's mask is stronger there. So
  /// where boxes overlap, the ink goes to the letter whose predicted outline
  /// covers it instead of to every box it lies in.
  ///
  /// [probabilities] holds one value per pixel of each letter's box, row by
  /// row (see `LetterMask.probabilities`).
  static List<List<List<int>>> extractWithMasks(img.Image image,
      List<List<int>> boxes, List<Float32List> probabilities,
      {double threshold = maskThreshold}) {
    double probabilityAt(int j, int x, int y) {
      final b = boxes[j];
      if (x < b[0] || x >= b[2] || y < b[1] || y >= b[3]) return 0;
      return probabilities[j][(y - b[1]) * (b[2] - b[0]) + (x - b[0])];
    }

    return [
      for (var i = 0; i < boxes.length; i++)
        _runs(image, boxes[i], (x, y) {
          final p = probabilityAt(i, x, y);
          if (p <= threshold) return false;
          for (var j = 0; j < boxes.length; j++) {
            if (j == i) continue;
            final other = probabilityAt(j, x, y);
            // On a tie the letter further left keeps the pixel.
            if (j < i ? other >= p : other > p) return false;
          }
          return true;
        })
    ];
  }

  static List<List<int>> _runs(
      img.Image image, List<int> bbox, bool Function(int x, int y)? keep) {
    final x1 = bbox[0].clamp(0, image.width);
    final y1 = bbox[1].clamp(0, image.height);
    final x2 = bbox[2].clamp(0, image.width);
    final y2 = bbox[3].clamp(0, image.height);
    if (x2 <= x1 || y2 <= y1) return [];
    final gray = img.grayscale(
        img.copyCrop(image, x: x1, y: y1, width: x2 - x1, height: y2 - y1));
    final threshold = ImagePreprocessor.otsuThreshold(gray).clamp(0, 200);
    final runs = <List<int>>[];
    for (var y = 0; y < gray.height; y++) {
      var start = -1;
      for (var x = 0; x <= gray.width; x++) {
        final ink = x < gray.width &&
            gray.getPixel(x, y).r <= threshold &&
            (keep == null || keep(x1 + x, y1 + y));
        if (ink && start < 0) start = x;
        if (!ink && start >= 0) {
          runs.add([x1 + start, y1 + y, x - start]);
          start = -1;
        }
      }
    }
    return runs;
  }
}
