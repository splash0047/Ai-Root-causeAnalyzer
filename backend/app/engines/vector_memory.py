"""
AI Root Cause Analyzer - Vector Memory Engine
Pinecone-based case memory for RCA pattern recognition.
Stores past diagnoses and finds similar historical cases to boost confidence.
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional


class VectorMemory:
    """
    Pinecone vector database integration for RCA case memory.
    - Stores each RCA diagnosis as a searchable vector
    - Retrieves similar past cases to boost diagnostic confidence
    - Uses integrated inference (server-side embedding)
    """

    def __init__(self, api_key: str = "", index_name: str = "rca-cases"):
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = "rca-diagnoses"
        self._index = None
        self._available = False
        self._init_client()

    def _init_client(self):
        """Initialize Pinecone client with integrated inference index."""
        if not self.api_key:
            print("[VectorMemory] No Pinecone API key provided. Memory disabled.")
            return

        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)

            # Check if index exists
            existing = [idx.name for idx in pc.list_indexes()]
            if self.index_name in existing:
                self._index = pc.Index(self.index_name)
                self._available = True
                print(f"[VectorMemory] Connected to Pinecone index: {self.index_name}")
            else:
                print(f"[VectorMemory] Index '{self.index_name}' not found. Memory disabled.")
        except Exception as e:
            print(f"[VectorMemory] Pinecone init failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def store_case(self, rca_id: int, rca_result: Dict[str, Any],
                   drift_report: Dict[str, Any]) -> bool:
        """
        Store an RCA diagnosis as a vector in Pinecone.
        Uses the diagnosis summary as the embedded text.
        """
        if not self._available:
            return False

        try:
            # Build the text to embed
            diagnosis_text = self._build_diagnosis_text(rca_result, drift_report)

            # Create a unique ID
            record_id = f"rca-{rca_id}-{hashlib.md5(diagnosis_text.encode()).hexdigest()[:8]}"

            # Metadata for filtering and display
            metadata = {
                "rca_id": rca_id,
                "root_cause": rca_result.get("root_cause", "Unknown")[:200],
                "severity": rca_result.get("severity", "unknown"),
                "confidence_score": rca_result.get("confidence_score", 0),
                "rca_mode": rca_result.get("rca_mode", "deep"),
                "timestamp": datetime.utcnow().isoformat(),
                "drifted_features": json.dumps(
                    drift_report.get("drifted_features", [])[:5]
                ),
                "top_feature": (
                    rca_result.get("ranked_features", [{}])[0].get("feature", "")
                    if rca_result.get("ranked_features") else ""
                ),
            }

            # Upsert with integrated inference (Pinecone embeds the text)
            self._index.upsert(
                vectors=[{
                    "id": record_id,
                    "metadata": metadata,
                    "values": [],  # Will be populated by integrated inference
                }],
                namespace=self.namespace,
            )

            # Also upsert via the records API for integrated inference
            self._upsert_with_inference(record_id, diagnosis_text, metadata)

            print(f"[VectorMemory] Stored case: {record_id}")
            return True

        except Exception as e:
            print(f"[VectorMemory] Failed to store case: {e}")
            return False

    def search_similar(self, rca_result: Dict[str, Any],
                       drift_report: Dict[str, Any],
                       top_k: int = 3) -> Dict[str, Any]:
        """
        Search for similar past RCA cases.

        Returns:
            Dict with 'match_score' (0-1), 'similar_cases' list, 'summary'
        """
        if not self._available:
            return {"match_score": 0.0, "similar_cases": [], "summary": "Vector memory unavailable"}

        try:
            diagnosis_text = self._build_diagnosis_text(rca_result, drift_report)

            # Search using integrated inference
            results = self._search_with_inference(diagnosis_text, top_k)

            if not results:
                return {
                    "match_score": 0.0,
                    "similar_cases": [],
                    "summary": "No similar past cases found",
                }

            # Extract matches
            similar_cases = []
            max_score = 0.0

            for match in results:
                score = match.get("score", 0)
                meta = match.get("metadata", {})
                max_score = max(max_score, score)

                similar_cases.append({
                    "rca_id": meta.get("rca_id", "unknown"),
                    "root_cause": meta.get("root_cause", "Unknown"),
                    "severity": meta.get("severity", "unknown"),
                    "confidence_score": meta.get("confidence_score", 0),
                    "similarity_score": round(score, 4),
                    "timestamp": meta.get("timestamp", ""),
                    "top_feature": meta.get("top_feature", ""),
                })

            # Normalize match score to 0-1 range
            # Scores > 0.7 indicate strong similarity
            normalized_score = min(1.0, max(0.0, (max_score - 0.3) / 0.7))

            summary = (
                f"Found {len(similar_cases)} similar case(s). "
                f"Best match: '{similar_cases[0]['root_cause']}' "
                f"(similarity: {max_score:.2%})"
            ) if similar_cases else "No similar cases found"

            return {
                "match_score": round(normalized_score, 4),
                "similar_cases": similar_cases,
                "summary": summary,
            }

        except Exception as e:
            print(f"[VectorMemory] Search failed: {e}")
            return {"match_score": 0.0, "similar_cases": [], "summary": f"Search error: {str(e)}"}

    def _build_diagnosis_text(self, rca_result: Dict, drift_report: Dict) -> str:
        """Build a comprehensive text representation of the RCA diagnosis for embedding."""
        parts = []

        # Root cause
        parts.append(f"Root Cause: {rca_result.get('root_cause', 'Unknown')}")
        parts.append(f"Detail: {rca_result.get('root_cause_detail', '')}")
        parts.append(f"Severity: {rca_result.get('severity', 'unknown')}")

        # Top features
        for feat in rca_result.get("ranked_features", [])[:3]:
            causal = "causally confirmed" if feat.get("causality_confirmed") else "statistical"
            parts.append(f"Feature: {feat['feature']} (impact: {feat['impact']:.4f}, {causal})")

        # Drift info
        drifted = drift_report.get("drifted_features", [])
        if drifted:
            parts.append(f"Drifted features: {', '.join(drifted)}")
            parts.append(f"Drift severity: {drift_report.get('overall_drift_severity', 'none')}")

        return " | ".join(parts)

    def _upsert_with_inference(self, record_id: str, text: str, metadata: Dict):
        """Upsert using Pinecone's integrated inference for embedding."""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)
            idx = pc.Index(self.index_name)

            # Use the upsert_records method for integrated inference
            record = {
                "_id": record_id,
                "diagnosis_text": text,  # This is the field mapped for embedding
                **{k: v for k, v in metadata.items()},
            }

            idx.upsert_records(
                namespace=self.namespace,
                records=[record],
            )
        except Exception as e:
            print(f"[VectorMemory] Inference upsert fallback: {e}")

    def _search_with_inference(self, query_text: str, top_k: int) -> List[Dict]:
        """Search using Pinecone's integrated inference for query embedding."""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)
            idx = pc.Index(self.index_name)

            # Use search with integrated inference
            results = idx.search(
                namespace=self.namespace,
                query={
                    "inputs": {"text": query_text},
                    "top_k": top_k,
                },
            )

            # Parse results
            matches = []
            if hasattr(results, "result") and hasattr(results.result, "hits"):
                for hit in results.result.hits:
                    matches.append({
                        "id": hit.get("_id", ""),
                        "score": hit.get("_score", 0),
                        "metadata": hit.get("fields", {}),
                    })
            elif isinstance(results, dict) and "result" in results:
                for hit in results["result"].get("hits", []):
                    matches.append({
                        "id": hit.get("_id", ""),
                        "score": hit.get("_score", 0),
                        "metadata": hit.get("fields", {}),
                    })

            return matches

        except Exception as e:
            print(f"[VectorMemory] Inference search error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._available:
            return {"available": False}

        try:
            stats = self._index.describe_index_stats()
            return {
                "available": True,
                "total_vectors": stats.get("total_vector_count", 0),
                "namespaces": stats.get("namespaces", {}),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
