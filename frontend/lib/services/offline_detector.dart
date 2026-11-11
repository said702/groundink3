import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import '../models/word_box.dart';
import 'onnx_inference.dart';
import 'image_preprocessor.dart';
import 'progress_notifier.dart';
import 'character_ink.dart';
import 'ink_color.dart';
import 'spelling/language_model.dart';
import 'spelling/spell_checker.dart';

/// Orders detected words in the same top-to-bottom, left-to-right order in
/// which a reader sees them.
List<WordBox> sortWordsInReadingOrder(List<WordBox> words) {
  final withBoxes = words.where((word) => word.bbox.length >= 4).toList();
  final withoutBoxes = words.where((word) => word.bbox.length < 4).toList();
  withBoxes.sort((a, b) => a.bbox[1].compareTo(b.bbox[1]));

  final rows = <List<WordBox>>[];
  final rowCenters = <double>[];
  final rowHeights = <double>[];
  for (final word in withBoxes) {
    final top = word.bbox[1].toDouble();
    final bottom = word.bbox[3].toDouble();
    final center = (top + bottom) / 2;
    final height = (bottom - top).abs();
    var row = -1;
    for (var i = 0; i < rows.length; i++) {
      final tolerance = math.max(12.0, math.max(height, rowHeights[i]) * 0.6);
      if ((center - rowCenters[i]).abs() <= tolerance) {
        row = i;
        break;
      }
    }
    if (row < 0) {
      rows.add([word]);
      rowCenters.add(center);
      rowHeights.add(height);
    } else {
      rows[row].add(word);
      rowCenters[row] = (rowCenters[row] * (rows[row].length - 1) + center) /
          rows[row].length;
      rowHeights[row] = (rowHeights[row] * (rows[row].length - 1) + height) /
          rows[row].length;
    }
  }

  final rowOrder = List.generate(rows.length, (index) => index)
    ..sort((a, b) => rowCenters[a].compareTo(rowCenters[b]));
  return [
    for (final row in rowOrder)
      ...(rows[row]..sort((a, b) => a.bbox[0].compareTo(b.bbox[0]))),
    ...withoutBoxes,
  ];
}

/// Checks one recognized word: (is correct, ranked suggestions).
///
/// [previous] and [next] are the neighboring words in reading order; they
/// rank the suggestions by context. [confidences] are the recognizer's
/// per-letter confidences, if known.
(bool, List<String>) checkSpelling(
  SpellChecker checker,
  String text, {
  String? previous,
  String? next,
  List<double>? confidences,
}) {
  if (checker.isCorrect(text)) return (true, const []);
  final suggestions = checker.suggest(text,
      confidences: confidences, previous: previous, next: next);
  return (false, [for (final s in suggestions) s.word]);
}

/// Spell-checks [words] (in reading order), using each word's neighbors as
/// context for the suggestions.
List<WordBox> checkWordsInContext(List<WordBox> words, SpellChecker checker) {
  final checked = <WordBox>[];
  for (var i = 0; i < words.length; i++) {
    final word = words[i];
    final confidences = [for (final c in word.characters) c.confidence];
    final (isCorrect, suggestions) = checkSpelling(
      checker,
      word.text,
      previous: i > 0 ? words[i - 1].text : null,
      next: i + 1 < words.length ? words[i + 1].text : null,
      confidences: confidences.length == word.text.length ? confidences : null,
    );
    checked.add(word.copyWith(isCorrect: isCorrect, suggestions: suggestions));
  }
  return checked;
}

/// Offline-Erkennungspipeline.
/// Führt Wort-Detection, Character-Recognition und Spellcheck
/// komplett lokal auf dem Gerät durch (kein Server nötig).
///
/// Die schwere ONNX-Inferenz läuft in einem separaten Isolate,
/// damit der Main-Thread (UI) nicht blockiert wird.
class OfflineDetector {
  /// Optionaler Progress-Notifier für UI-Updates.
  static ProgressNotifier? progressNotifier;
  static Uint8List? _wordModel;
  static Uint8List? _charModel;
  static Uint8List? _spellingModel;

  /// Long-lived worker isolate holding the ONNX sessions and the spelling
  /// model.
  static Future<(Isolate, SendPort)>? _worker;
  static const _workerExited = 'worker-exited';

  /// Spell checker of the worker isolate (null on the UI isolate).
  static SpellChecker? _checker;

  /// Loads asset data once and starts the worker isolate in the background,
  /// so the first photo does not pay for creating the sessions.
  static Future<void> initOnnx() async {
    _wordModel ??= Uint8List.sublistView(
        await rootBundle.load('assets/models/word_detection_model.onnx'));
    _charModel ??= Uint8List.sublistView(
        await rootBundle.load('assets/models/char_model_nano_256.onnx'));
    _spellingModel ??= Uint8List.sublistView(
        await rootBundle.load('assets/models/english_lm.bin.gz'));
    _ensureWorker().ignore();
  }

  static Future<(Isolate, SendPort)> _ensureWorker() {
    final existing = _worker;
    if (existing != null) return existing;
    final spawning = _spawnWorker();
    _worker = spawning;
    // A failed start must not be cached; the next photo retries.
    spawning.then((_) {}, onError: (Object _) {
      if (identical(_worker, spawning)) _worker = null;
    });
    return spawning;
  }

  static Future<(Isolate, SendPort)> _spawnWorker() async {
    final ready = ReceivePort();
    final exit = ReceivePort();
    final isolate = await Isolate.spawn(
      _workerMain,
      (ready.sendPort, _wordModel!, _charModel!, _spellingModel!),
      onExit: exit.sendPort,
      debugName: 'OfflineDetector',
    );
    final current = _worker;
    exit.first.then((_) {
      exit.close();
      if (identical(_worker, current)) _worker = null;
    });
    final first = await ready.first;
    ready.close();
    if (first is _WorkerFailure) first.rethrowIt();
    return (isolate, first as SendPort);
  }

  /// Entry point of the worker isolate: creates the sessions and the spell
  /// checker once, then answers one request at a time, in order.
  static Future<void> _workerMain(
      (SendPort, Uint8List, Uint8List, Uint8List) args) async {
    final (ready, wordModel, charModel, spellingModel) = args;
    final requests = ReceivePort();
    try {
      await OnnxInference.init(wordModel: wordModel, charModel: charModel);
      _checker = SpellChecker(LanguageModel.fromGzip(spellingModel));
    } catch (e, st) {
      requests.close();
      OnnxInference.dispose();
      ready.send(_WorkerFailure(e, st.toString()));
      return;
    }
    ready.send(requests.sendPort);
    await for (final message in requests) {
      final request = message as _Request;
      try {
        request.reply.send(await request.run());
      } catch (e, st) {
        try {
          request.reply.send(_WorkerFailure(e, st.toString()));
        } catch (_) {
          // The error object itself cannot cross isolates; keep its text.
          request.reply
              .send(_WorkerFailure(Exception(e.toString()), st.toString()));
        }
      }
    }
  }

  /// Sends [request] to the worker and returns its answer.
  static Future<Object?> _ask(_Request Function(SendPort reply) request) async {
    final (isolate, port) = await _ensureWorker();
    final reply = ReceivePort();
    isolate.addOnExitListener(reply.sendPort, response: _workerExited);
    port.send(request(reply.sendPort));
    final response = await reply.first;
    isolate.removeOnExitListener(reply.sendPort);
    reply.close();
    if (response == _workerExited) {
      throw Exception('The recognition worker stopped unexpectedly.');
    }
    if (response is _WorkerFailure) response.rethrowIt();
    return response;
  }

  /// Haupt-Pipeline: Foto → erkannte Wörter mit Spellcheck.
  /// Mit optionalem Progress-Callback für interaktive UI.
  ///
  /// Die komplette schwere Berechnung läuft im Worker-Isolate.
  static Future<List<WordBox>> detect(File imageFile) async {
    final progressPort = ReceivePort();
    final subscription = progressPort.listen((message) {
      if (message is DetectionProgress) _updateProgress(message);
    });
    try {
      // Asset access must stay on the main isolate.
      await initOnnx();

      // 2. Bild laden (im Main-Isolate, für Progress-UI)
      _updateProgress(DetectionProgress.loadingImage);
      final bytes = await imageFile.readAsBytes();

      // 3. Komplette Pipeline im Worker-Isolate ausführen
      final results = await _ask((reply) =>
          _DetectRequest(bytes, reply, progressPort.sendPort)) as List<WordBox>;

      _updateProgress(DetectionProgress.done);
      return results;
    } catch (e) {
      _updateProgress(DetectionProgress.error(e.toString()));
      rethrow;
    } finally {
      await subscription.cancel();
      progressPort.close();
    }
  }

  /// Spell-checks a word the user corrected: (is correct, suggestions).
  /// [previous] and [next] are its neighbors in reading order.
  static Future<(bool, List<String>)> checkWord(String text,
      {String? previous, String? next}) async {
    await initOnnx();
    final response = await _ask(
        (reply) => _SpellRequest(text, previous, next, reply));
    return response as (bool, List<String>);
  }

  /// Interne Pipeline-Funktion, die im Isolate läuft.
  /// Kein Zugriff auf ProgressNotifier (nicht über Isolate-Grenzen teilbar).
  static Future<List<WordBox>> _detectInIsolate(
      Uint8List bytes, SendPort progress) async {
    // Bild dekodieren
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception('The image could not be decoded.');
    }

    // Wort-Detection (YOLO26n ONNX)
    progress.send(DetectionProgress.detectingWords);
    final wordResults = OnnxInference.detectWords(image, confThreshold: 0.25);
    if (wordResults.isEmpty) {
      return [];
    }

    // Pro Wort: Character-Recognition
    final results = <WordBox>[];

    for (int i = 0; i < wordResults.length; i++) {
      progress.send(DetectionProgress.recognizingCharacter(
          current: i + 1, total: wordResults.length));
      final w = wordResults[i];
      final bbox = (w['bbox'] as List).cast<int>();
      final conf = w['confidence'] as double;

      // Wort-Crop aus dem Originalbild
      final x1 = bbox[0].clamp(0, image.width).toInt();
      final y1 = bbox[1].clamp(0, image.height).toInt();
      final x2 = bbox[2].clamp(0, image.width).toInt();
      final y2 = bbox[3].clamp(0, image.height).toInt();
      if (x2 <= x1 || y2 <= y1) continue;
      final cropW = (x2 - x1).clamp(1, image.width - x1);
      final cropH = (y2 - y1).clamp(1, image.height - y1);
      final crop =
          img.copyCrop(image, x: x1, y: y1, width: cropW, height: cropH);
      if (crop.width == 0 || crop.height == 0) continue;

      // Preprocessing für das Zeichenmodell
      final (preprocessed, offsetX, offsetY) =
          ImagePreprocessor.preprocessWithOffset(crop);

      // Zeichen erkennen (mit Preprocessing)
      final charResults = OnnxInference.recognizeCharacters(
        preprocessed,
        [x1 + offsetX, y1 + offsetY, x2, y2],
      );

      // Wort-Text aus Buchstaben (already left to right; see
      // CharacterDecoder for the order of equal positions).
      final sortedChars = charResults;

      final text = sortedChars.map((c) => c['char'] as String).join().trim();
      if (text.isEmpty) continue;

      // Ink of each letter: with the predicted letter outlines when the
      // model returns them, otherwise all dark pixels in the letter's box.
      final boxes = [
        for (final c in sortedChars) (c['bbox'] as List).cast<int>()
      ];
      final masks = [
        for (final c in sortedChars) c['maskProbabilities'] as Float32List?
      ];
      final inks = masks.every((m) => m != null)
          ? CharacterInk.extractWithMasks(
              image, boxes, [for (final m in masks) m!])
          : [for (final b in boxes) CharacterInk.extract(image, b)];

      var index = 0;
      final characters = sortedChars
          .map((c) => CharacterResult(
                char: c['char'] as String,
                bbox: (c['bbox'] as List).cast<int>(),
                contour: (c['contour'] as List).cast<List<int>>(),
                confidence: c['confidence'] as double,
                inkRuns: inks[index++],
                mask: c['mask'] != null
                    ? (c['mask'] as List)
                        .map((row) => (row as List)
                            .map((e) => (e as num).toDouble())
                            .toList())
                        .toList()
                    : null,
                maskW: c['maskW'] as int?,
                maskH: c['maskH'] as int?,
              ))
          .toList();

      results.add(WordBox(
        text: text,
        bbox: [x1, y1, x2, y2],
        confidence: conf,
        isCorrect: true, // set by the spell check below
        suggestions: const [],
        characters: characters,
        inkColor: InkColor.measure(
            image, [for (final c in characters) ...c.inkRuns], [x1, y1, x2, y2]),
      ));
    }

    // Spellcheck (offline) in reading order, so that every word's
    // neighbors can rank its suggestions.
    return checkWordsInContext(sortWordsInReadingOrder(results), _checker!);
  }

  /// Interne Methode für Progress-Updates.
  static void _updateProgress(DetectionProgress progress) {
    progressNotifier?.update(progress);
  }
}

/// A job for the worker isolate.
sealed class _Request {
  final SendPort reply;
  const _Request(this.reply);
  Future<Object?> run();
}

class _DetectRequest extends _Request {
  final Uint8List bytes;
  final SendPort progress;
  const _DetectRequest(this.bytes, super.reply, this.progress);

  @override
  Future<Object?> run() => OfflineDetector._detectInIsolate(bytes, progress);
}

class _SpellRequest extends _Request {
  final String text;
  final String? previous;
  final String? next;
  const _SpellRequest(this.text, this.previous, this.next, SendPort reply)
      : super(reply);

  @override
  Future<Object?> run() async => checkSpelling(OfflineDetector._checker!, text,
      previous: previous, next: next);
}

/// An error raised in the worker isolate, rethrown on the caller's side.
class _WorkerFailure {
  final Object error;
  final String stackTrace;

  const _WorkerFailure(this.error, this.stackTrace);

  Never rethrowIt() =>
      Error.throwWithStackTrace(error, StackTrace.fromString(stackTrace));
}
