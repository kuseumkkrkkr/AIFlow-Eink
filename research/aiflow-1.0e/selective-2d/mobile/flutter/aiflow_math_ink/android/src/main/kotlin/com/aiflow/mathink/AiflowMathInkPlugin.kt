package com.aiflow.mathink

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.security.MessageDigest

/**
 * Contract-first bridge.  The deployable ONNX engine is injected by the host
 * AAR; until model parity is admitted, this class returns an immutable raw
 * fallback rather than pretending that a recognition result exists.
 */
class AiflowMathInkPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel
    private val sessions = mutableMapOf<String, Session>()

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "aiflow/math_ink/v1")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        sessions.clear()
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            val args = call.arguments<Map<String, Any?>>() ?: emptyMap()
            val formulaId = args["formulaId"] as? String ?: throw IllegalArgumentException("formulaId is required")
            when (call.method) {
                "startSession" -> {
                    sessions[formulaId] = Session(formulaId)
                    result.success(mapOf("schema" to SCHEMA, "formula_id" to formulaId, "revision" to 0))
                }
                "strokeEnd" -> result.success(session(formulaId).strokeEnd(args))
                "replaceInk" -> result.success(session(formulaId).replaceInk(args))
                "complete" -> result.success(session(formulaId).complete(args))
                "disposeSession" -> {
                    sessions.remove(formulaId)
                    result.success(mapOf("schema" to SCHEMA, "formula_id" to formulaId, "disposed" to true))
                }
                else -> result.notImplemented()
            }
        } catch (error: IllegalArgumentException) {
            result.error("invalid_argument", error.message, null)
        }
    }

    private fun session(formulaId: String) = sessions[formulaId]
        ?: throw IllegalArgumentException("unknown formula session")

    private class Session(private val formulaId: String) {
        private var revision = 0
        private var strokes: List<Map<String, Any?>> = emptyList()
        private val completed = mutableMapOf<String, Map<String, Any?>>()

        fun strokeEnd(args: Map<String, Any?>): Map<String, Any?> {
            val next = integer(args, "revision")
            require(next == revision + 1) { "revision must advance by one" }
            val stroke = map(args, "stroke")
            require(integer(stroke, "order") == strokes.size) { "stroke order must be contiguous" }
            require((stroke["points"] as? List<*>)?.isNotEmpty() == true) { "stroke requires points" }
            strokes = strokes + stroke
            revision = next
            completed.clear()
            return update("fast")
        }

        fun replaceInk(args: Map<String, Any?>): Map<String, Any?> {
            val next = integer(args, "revision")
            require(next > revision) { "replacement revision must advance" }
            val replacement = (args["strokes"] as? List<*>)?.mapIndexed { index, value ->
                val stroke = value as? Map<String, Any?> ?: throw IllegalArgumentException("invalid stroke")
                require(integer(stroke, "order") == index) { "stroke order must be contiguous" }
                require((stroke["points"] as? List<*>)?.isNotEmpty() == true) { "stroke requires points" }
                stroke
            } ?: throw IllegalArgumentException("strokes is required")
            strokes = replacement
            revision = next
            completed.clear()
            return update("fast") + mapOf("event" to "replace_ink")
        }

        fun complete(args: Map<String, Any?>): Map<String, Any?> {
            require(integer(args, "revision") == revision) { "stale completion revision" }
            val eventId = args["completionEventId"] as? String ?: throw IllegalArgumentException("completionEventId is required")
            val rawSha = rawFallback()["canonical_sha256"] as String
            return completed[rawSha]?.plus(mapOf(
                "completion_event_id" to eventId, "idempotent_replay" to true,
            )) ?: update("final", eventId).also { completed[rawSha] = it }
        }

        private fun update(stage: String, completionEventId: String? = null): Map<String, Any?> {
            val raw = rawFallback()
            return mapOf(
                "schema" to RESULT_SCHEMA, "formula_id" to formulaId, "revision" to revision,
                "stage" to stage, "route" to "fallback", "committed" to (stage == "final"),
                "groups" to emptyList<Any>(), "symbols" to emptyList<Any>(), "formula_latex" to null,
                "raw_sha256" to raw["canonical_sha256"], "raw_fallback" to raw,
                "failure_reason" to "onnx_engine_not_admitted", "completion_event_id" to completionEventId,
                "timings_ms" to emptyMap<String, Any>(), "region_strokes" to emptyList<Int>(),
                "candidate_count" to 0, "winner_reason" to "fallback",
                "idempotent_replay" to false,
            )
        }

        private fun rawFallback(): Map<String, Any?> {
            val canonical = AiflowMathInkPlugin.canonical(strokes).toByteArray(Charsets.UTF_8)
            val hash = MessageDigest.getInstance("SHA-256").digest(canonical).joinToString("") { "%02x".format(it) }
            val points = strokes.sumOf { (it["points"] as? List<*>)?.size ?: 0 }
            return mapOf("schema" to RAW_SCHEMA, "formula_id" to formulaId, "strokes" to strokes,
                "stroke_count" to strokes.size, "point_count" to points, "canonical_sha256" to hash,
                "all_input_strokes_preserved" to true)
        }
    }

    companion object {
        private const val SCHEMA = "aiflow-revisioned-formula-session/v1"
        private const val RESULT_SCHEMA = "aiflow-recognition-update/v1"
        private const val RAW_SCHEMA = "aiflow-raw-ink-fallback/v1"
        private fun map(values: Map<String, Any?>, key: String): Map<String, Any?> =
            values[key] as? Map<String, Any?> ?: throw IllegalArgumentException("$key is required")
        private fun integer(values: Map<String, Any?>, key: String): Int =
            (values[key] as? Number)?.toInt() ?: throw IllegalArgumentException("$key is required")
        private fun canonical(value: Any?): String = when (value) {
            null -> "null"
            is String -> "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
            is Number, is Boolean -> value.toString()
            is List<*> -> value.joinToString(prefix = "[", postfix = "]") { canonical(it) }
            is Map<*, *> -> value.entries.sortedBy { it.key.toString() }.joinToString(prefix = "{", postfix = "}") {
                canonical(it.key.toString()) + ":" + canonical(it.value)
            }
            else -> throw IllegalArgumentException("unsupported raw ink value")
        }
    }
}
