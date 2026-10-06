import 'package:flutter/services.dart';

/// Android bridge. All stroke points are sent unchanged, including one-point
/// strokes; rendering/OCR serializers must not be used for this payload.
class AiflowMathInk {
  AiflowMathInk({MethodChannel? channel})
    : _channel = channel ?? const MethodChannel('aiflow/math_ink/v1');

  final MethodChannel _channel;

  Future<Map<String, dynamic>> startSession(String formulaId) =>
      _invoke('startSession', {'formulaId': formulaId});

  Future<Map<String, dynamic>> strokeEnd({
    required String formulaId,
    required int revision,
    required Map<String, dynamic> stroke,
  }) => _invoke('strokeEnd', {
    'formulaId': formulaId,
    'revision': revision,
    'stroke': stroke,
  });

  Future<Map<String, dynamic>> replaceInk({
    required String formulaId,
    required int revision,
    required List<Map<String, dynamic>> strokes,
  }) => _invoke('replaceInk', {
    'formulaId': formulaId,
    'revision': revision,
    'strokes': strokes,
  });

  Future<Map<String, dynamic>> complete({
    required String formulaId,
    required int revision,
    required String completionEventId,
  }) => _invoke('complete', {
    'formulaId': formulaId,
    'revision': revision,
    'completionEventId': completionEventId,
  });

  Future<Map<String, dynamic>> disposeSession(String formulaId) =>
      _invoke('disposeSession', {'formulaId': formulaId});

  Future<Map<String, dynamic>> _invoke(
    String method,
    Map<String, dynamic> arguments,
  ) async {
    final value = await _channel.invokeMapMethod<String, dynamic>(
      method,
      arguments,
    );
    if (value == null) {
      throw PlatformException(
        code: 'empty_result',
        message: '$method returned no result',
      );
    }
    return value;
  }
}
