"""
AegisVisionEnv – OpenEnv-compatible AI training environment
for women's digital safety and image authenticity detection.
"""

import random
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
import time


class Action(str, Enum):
    MARK_REAL = "MARK_REAL"
    MARK_FAKE = "MARK_FAKE"
    APPLY_PROTECTION = "APPLY_PROTECTION"
    GENERATE_REPORT = "GENERATE_REPORT"
    IGNORE = "IGNORE"


class ImageType(str, Enum):
    REAL = "real"
    MORPHED = "morphed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


PLATFORMS = [
    "Instagram", "WhatsApp", "Telegram", "Twitter/X",
    "Snapchat", "Facebook", "LinkedIn", "Discord", "TikTok"
]

COMPRESSION_TYPES = ["JPEG", "WebP", "HEIC", "PNG", "H.264"]

SCENARIO_CONTEXTS = [
    "Profile photo submitted for verification",
    "Image shared in private group",
    "Photo uploaded to dating platform",
    "Image detected in harassment chain",
    "Avatar used in social engineering attempt",
    "Media in workplace communication",
    "Image flagged by community report",
]


@dataclass
class ImageScenario:
    scenario_id: str
    image_type: str
    platform: str
    compression: str
    risk_level: str
    context: str
    manipulation_hints: list
    timestamp: float

    def to_dict(self):
        return asdict(self)


@dataclass
class EnvState:
    scenario: Optional[dict]
    score: int
    step_count: int
    correct_detections: int
    wrong_detections: int
    missed_fakes: int
    smart_actions: int
    threat_level: str
    last_action: Optional[str]
    last_reward: int
    last_feedback: str
    session_id: str
    is_done: bool

    def to_dict(self):
        return asdict(self)


class AegisVisionEnv:
    """
    OpenEnv-compatible environment for AI-powered image authenticity training.
    Implements reset/state/step interface per OpenEnv specification.
    """

    MAX_STEPS = 20

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self._state = self._blank_state()

    def _blank_state(self) -> EnvState:
        return EnvState(
            scenario=None,
            score=0,
            step_count=0,
            correct_detections=0,
            wrong_detections=0,
            missed_fakes=0,
            smart_actions=0,
            threat_level="NOMINAL",
            last_action=None,
            last_reward=0,
            last_feedback="Awaiting scenario initialization.",
            session_id=self.session_id,
            is_done=False,
        )

    def _generate_scenario(self) -> ImageScenario:
        image_type = random.choice(list(ImageType))
        risk_level = random.choices(
            list(RiskLevel),
            weights=[20, 35, 30, 15],  # weighted distribution
            k=1
        )[0]

        hints = []
        if image_type == ImageType.MORPHED:
            all_hints = [
                "Inconsistent lighting direction detected",
                "Facial boundary artifacts present",
                "Metadata timestamp anomaly",
                "GAN fingerprint pattern identified",
                "Compression ghost artifacts",
                "Eye reflection asymmetry",
                "Blending seam near hairline",
                "EXIF data stripped or altered",
            ]
            num_hints = random.randint(1, 4)
            hints = random.sample(all_hints, num_hints)
        else:
            hints = ["No manipulation artifacts detected"]

        return ImageScenario(
            scenario_id=str(uuid.uuid4()),
            image_type=image_type.value,
            platform=random.choice(PLATFORMS),
            compression=random.choice(COMPRESSION_TYPES),
            risk_level=risk_level.value,
            context=random.choice(SCENARIO_CONTEXTS),
            manipulation_hints=hints,
            timestamp=time.time(),
        )

    def _compute_threat_level(self) -> str:
        score = self._state.score
        if score >= 80:
            return "SECURE"
        elif score >= 40:
            return "GUARDED"
        elif score >= 0:
            return "ELEVATED"
        elif score >= -20:
            return "HIGH"
        else:
            return "CRITICAL"

    def reset(self) -> dict:
        """Reset environment, generate new scenario. Returns initial observation."""
        self.session_id = str(uuid.uuid4())
        self._state = self._blank_state()
        scenario = self._generate_scenario()
        self._state.scenario = scenario.to_dict()
        self._state.last_feedback = "New scenario loaded. Analyze and take action."

        return {
            "observation": self._state.to_dict(),
            "info": {
                "message": "Environment reset successfully.",
                "session_id": self.session_id,
                "env_version": "1.0.0",
            }
        }

    def state(self) -> dict:
        """Return current environment state."""
        if self._state.scenario is None:
            return {
                "observation": self._state.to_dict(),
                "info": {"message": "Environment not initialized. Call /reset first."}
            }
        return {
            "observation": self._state.to_dict(),
            "info": {"message": "State retrieved successfully."}
        }

    def step(self, action: str) -> dict:
        """
        Execute an action in the environment.
        Returns: observation, reward, done, info
        """
        if self._state.is_done:
            return {
                "observation": self._state.to_dict(),
                "reward": 0,
                "done": True,
                "info": {"message": "Episode complete. Call /reset to start new session."}
            }

        if self._state.scenario is None:
            return {
                "observation": self._state.to_dict(),
                "reward": 0,
                "done": False,
                "info": {"message": "No scenario loaded. Call /reset first."}
            }

        try:
            action_enum = Action(action.upper())
        except ValueError:
            return {
                "observation": self._state.to_dict(),
                "reward": -1,
                "done": False,
                "info": {"message": f"Invalid action '{action}'. Valid: {[a.value for a in Action]}"}
            }

        scenario = self._state.scenario
        actual_type = scenario["image_type"]
        reward = 0
        feedback = ""

        if action_enum == Action.MARK_REAL:
            if actual_type == ImageType.REAL.value:
                reward = 10
                self._state.correct_detections += 1
                feedback = "✅ CORRECT — Image authenticated as genuine. +10 pts"
            else:
                reward = -5
                self._state.wrong_detections += 1
                feedback = "❌ WRONG — This was a manipulated image. Threat missed. -5 pts"

        elif action_enum == Action.MARK_FAKE:
            if actual_type == ImageType.MORPHED.value:
                reward = 10
                self._state.correct_detections += 1
                feedback = "✅ CORRECT — Manipulation detected and flagged. +10 pts"
            else:
                reward = -5
                self._state.wrong_detections += 1
                feedback = "❌ WRONG — Image was authentic. False alarm raised. -5 pts"

        elif action_enum == Action.APPLY_PROTECTION:
            reward = 3
            self._state.smart_actions += 1
            feedback = "🛡️ SMART — Watermark protection applied to scenario image. +3 pts"

        elif action_enum == Action.GENERATE_REPORT:
            reward = 3
            self._state.smart_actions += 1
            feedback = "📋 SMART — Forensic report generated for evidence chain. +3 pts"

        elif action_enum == Action.IGNORE:
            if actual_type == ImageType.MORPHED.value:
                reward = -10
                self._state.missed_fakes += 1
                feedback = "⚠️ DANGER — Fake image ignored. Victim protection failed. -10 pts"
            else:
                reward = 0
                feedback = "➖ NEUTRAL — Real image bypassed. No action taken. 0 pts"

        self._state.score += reward
        self._state.last_reward = reward
        self._state.last_action = action_enum.value
        self._state.last_feedback = feedback
        self._state.step_count += 1
        self._state.threat_level = self._compute_threat_level()

        # Generate next scenario
        new_scenario = self._generate_scenario()
        self._state.scenario = new_scenario.to_dict()

        done = self._state.step_count >= self.MAX_STEPS
        self._state.is_done = done

        return {
            "observation": self._state.to_dict(),
            "reward": reward,
            "done": done,
            "info": {
                "message": feedback,
                "new_scenario_loaded": True,
                "steps_remaining": self.MAX_STEPS - self._state.step_count,
            }
        }

    def protect_image(self, image_name: str = "uploaded_image") -> dict:
        """Simulate watermarking / steganographic protection."""
        risk_reduction = random.randint(55, 92)
        watermark_strength = random.uniform(0.85, 0.99)
        return {
            "protected_image_url": f"/protected/{uuid.uuid4().hex[:12]}_{image_name}",
            "risk_reduction_percentage": risk_reduction,
            "watermark_strength": round(watermark_strength, 3),
            "method": random.choice(["DCT Steganography", "LSB Embedding", "Frequency Domain Watermark", "Perceptual Hash Binding"]),
            "timestamp": time.time(),
            "certificate_id": f"AEGIS-{uuid.uuid4().hex[:8].upper()}",
            "status": "PROTECTED",
        }

    def compare_images(self, original_name: str = "original", suspect_name: str = "suspect") -> dict:
        """Simulate forensic image comparison."""
        manipulation_score = random.uniform(0.05, 0.98)
        is_manipulated = manipulation_score > 0.45

        confidence = random.uniform(0.72, 0.99) if manipulation_score > 0.7 or manipulation_score < 0.15 else random.uniform(0.55, 0.80)

        regions = []
        if is_manipulated:
            region_count = random.randint(1, 4)
            labels = ["Face swap region", "Background splice", "Texture inconsistency", "Lighting anomaly", "Clone stamp area", "GAN artifact cluster"]
            for _ in range(region_count):
                regions.append({
                    "label": random.choice(labels),
                    "severity": random.choice(["low", "medium", "high", "critical"]),
                    "x": random.randint(10, 80),
                    "y": random.randint(10, 80),
                    "w": random.randint(10, 30),
                    "h": random.randint(10, 30),
                })

        return {
            "manipulation_score": round(manipulation_score, 4),
            "confidence": round(confidence, 4),
            "verdict": "MANIPULATED" if is_manipulated else "AUTHENTIC",
            "difference_map": f"/diff_maps/{uuid.uuid4().hex[:12]}.png",
            "heatmap_url": f"/heatmaps/{uuid.uuid4().hex[:12]}.png",
            "flagged_regions": regions,
            "analysis_method": random.choice(["ELA Analysis", "PRNU Fingerprinting", "GAN Detector", "Metadata Forensics", "DCT Coefficient Analysis"]),
            "risk_level": "HIGH" if manipulation_score > 0.7 else ("MEDIUM" if manipulation_score > 0.4 else "LOW"),
            "timestamp": time.time(),
        }

