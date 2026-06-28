"""
Thread-safe in-memory job store for async inspection jobs.
Each job tracks: id, status, timestamps, config snapshot, result, file paths.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Job status constants
PENDING    = "PENDING"
PROCESSING = "PROCESSING"
DONE       = "DONE"
FAILED     = "FAILED"


class Job:
    """Represents a single async inspection job."""
    def __init__(self, job_id: str, config_snapshot: Dict[str, Any]):
        self.job_id: str = job_id
        self.status: str = PENDING
        self.submitted_at: datetime = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self.config_snapshot: Dict[str, Any] = config_snapshot
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.processing_time_ms: Optional[int] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "objects": self.result.get("objects") if self.result else None,
            "fod_status": self.result.get("status") if self.result else None,
        }

    def to_status(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": self.processing_time_ms,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    """Async-safe in-memory registry for inspection jobs."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, config_snapshot: Dict[str, Any]) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, config_snapshot=config_snapshot)
        async with self._lock:
            self._jobs[job_id] = job
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_all(self, status_filter: Optional[str] = None) -> list[Job]:
        async with self._lock:
            jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        # Sort newest first
        return sorted(jobs, key=lambda j: j.submitted_at, reverse=True)

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
        return False

    async def purge_old(self, max_age_seconds: int) -> int:
        """Delete DONE/FAILED jobs older than max_age_seconds. Returns count deleted."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        to_delete = []
        async with self._lock:
            for jid, job in self._jobs.items():
                if job.status in (DONE, FAILED) and job.submitted_at < cutoff:
                    to_delete.append(jid)
            for jid in to_delete:
                del self._jobs[jid]
        return len(to_delete)

    async def mark_processing(self, job_id: str):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = PROCESSING

    async def mark_done(self, job_id: str, result: Dict[str, Any], processing_time_ms: int):
        async with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = DONE
                job.result = result
                job.processing_time_ms = processing_time_ms
                job.completed_at = datetime.now(timezone.utc)

    async def mark_failed(self, job_id: str, error: str):
        async with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = FAILED
                job.error = error
                job.completed_at = datetime.now(timezone.utc)


# Singleton instance shared across the app
job_store = JobStore()
