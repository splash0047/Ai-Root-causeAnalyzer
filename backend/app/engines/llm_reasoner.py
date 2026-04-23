"""
AI Root Cause Analyzer - LLM Reasoning Engine
Generates human-readable explanations and fix suggestions using Gemini (primary)
with OpenAI fallback. Structured prompt engineering with full diagnostic context.
"""
import json
from typing import Dict, Any, Optional


class LLMReasoner:
    """
    Uses LLM to generate natural-language RCA explanations and suggested fixes.
    Primary: Google Gemini | Fallback: OpenAI GPT-4o-mini
    """

    def __init__(self, gemini_api_key: str = "", openai_api_key: str = ""):
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self._gemini_model = None
        self._openai_client = None

    def _init_gemini(self):
        """Lazy-initialize Gemini client."""
        if self._gemini_model is None and self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            except Exception as e:
                print(f"[LLMReasoner] Gemini init failed: {e}")

    def _init_openai(self):
        """Lazy-initialize OpenAI client."""
        if self._openai_client is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"[LLMReasoner] OpenAI init failed: {e}")

    def generate_explanation(
        self,
        rca_result: Dict[str, Any],
        drift_report: Dict[str, Any],
        integrity_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a structured LLM explanation from the full RCA context.

        Returns:
            Dict with 'explanation', 'suggested_fix', 'confidence_reasoning', 'provider'
        """
        prompt = self._build_prompt(rca_result, drift_report, integrity_report)

        # Try Gemini first
        result = self._call_gemini(prompt)
        if result:
            result["provider"] = "gemini"
            return result

        # Fallback to OpenAI
        result = self._call_openai(prompt)
        if result:
            result["provider"] = "openai"
            return result

        # Both failed - return structured fallback
        return {
            "explanation": rca_result.get("root_cause_detail", "Analysis complete."),
            "suggested_fix": "Review the ranked features and reasoning chain for manual diagnosis.",
            "confidence_reasoning": "LLM unavailable. Confidence based on statistical signals only.",
            "provider": "fallback",
        }

    def _build_prompt(self, rca_result: Dict, drift_report: Dict,
                      integrity_report: Dict) -> str:
        """Build a structured prompt with full diagnostic context."""
        # Extract key information
        root_cause = rca_result.get("root_cause", "Unknown")
        confidence = rca_result.get("confidence_score", 0)
        severity = rca_result.get("severity", "unknown")
        ranked_features = rca_result.get("ranked_features", [])
        reasoning_chain = rca_result.get("reasoning_chain", [])

        # Format ranked features
        features_text = ""
        for i, feat in enumerate(ranked_features[:5], 1):
            causal = " ✓ CAUSALLY CONFIRMED" if feat.get("causality_confirmed") else ""
            features_text += f"  {i}. {feat['feature']} (impact: {feat['impact']:.4f}, source: {feat['source']}{causal})\n"

        # Format reasoning chain
        chain_text = ""
        for step in reasoning_chain:
            chain_text += f"  - {step['step']}: {step['result']} (score: {step.get('score', 'N/A')})\n"

        # Drift details
        drifted = drift_report.get("drifted_features", [])
        drift_severity = drift_report.get("overall_drift_severity", "none")

        # Integrity issues
        integrity_issues = integrity_report.get("issue_summary", [])

        prompt = f"""You are an expert ML systems diagnostician. Analyze the following Root Cause Analysis results and provide:
1. A clear, actionable EXPLANATION of what went wrong and why
2. A specific SUGGESTED FIX with concrete steps
3. CONFIDENCE REASONING explaining why the confidence score is what it is

--- RCA DIAGNOSTIC CONTEXT ---

ROOT CAUSE: {root_cause}
CONFIDENCE: {confidence:.2%}
SEVERITY: {severity}

RANKED FEATURES BY IMPACT:
{features_text if features_text else "  No features ranked."}

REASONING CHAIN:
{chain_text if chain_text else "  No reasoning chain available."}

DRIFT STATUS: {drift_severity} severity
DRIFTED FEATURES: {', '.join(drifted) if drifted else 'None'}

DATA INTEGRITY ISSUES: {'; '.join(integrity_issues) if integrity_issues else 'None'}

--- RESPONSE FORMAT ---
Respond in EXACTLY this JSON format (no markdown, no code blocks):
{{
  "explanation": "<2-3 paragraph technical explanation of the root cause, referencing specific features and their causal relationships>",
  "suggested_fix": "<Numbered list of concrete remediation steps>",
  "confidence_reasoning": "<1 paragraph explaining the confidence score based on signal convergence>"
}}"""

        return prompt

    def _call_gemini(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Gemini API and parse structured response."""
        self._init_gemini()
        if not self._gemini_model:
            return None

        try:
            response = self._gemini_model.generate_content(prompt)
            text = response.text.strip()
            return self._parse_llm_response(text)
        except Exception as e:
            print(f"[LLMReasoner] Gemini call failed: {e}")
            return None

    def _call_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI API and parse structured response."""
        self._init_openai()
        if not self._openai_client:
            return None

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an ML systems diagnostician. Always respond in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_llm_response(text)
        except Exception as e:
            print(f"[LLMReasoner] OpenAI call failed: {e}")
            return None

    def _parse_llm_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response, handling potential markdown wrapping."""
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            parsed = json.loads(text)
            return {
                "explanation": parsed.get("explanation", ""),
                "suggested_fix": parsed.get("suggested_fix", ""),
                "confidence_reasoning": parsed.get("confidence_reasoning", ""),
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, treat the whole response as the explanation
            return {
                "explanation": text[:2000],
                "suggested_fix": "Unable to parse structured fix. Review raw explanation.",
                "confidence_reasoning": "Response was not in expected format.",
            }
