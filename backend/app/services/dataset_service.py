from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import CallRecord, Dataset


def _normalize_manifest(data: dict | list) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        rows: list[dict] = []
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("callLogId", value.get("callLogId") or key)
            rows.append(row)
        return rows
    raise ValueError("Manifest must be a JSON object or array")


async def import_manifest_file(
    session: AsyncSession,
    *,
    name: str,
    file_path: Path,
    description: str | None = None,
) -> Dataset:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    rows = _normalize_manifest(payload)

    existing = await session.scalar(select(Dataset).where(Dataset.name == name))
    if existing:
        dataset = existing
        dataset.description = description
        dataset.source_path = str(file_path)
        for call in list(dataset.calls):
            await session.delete(call)
    else:
        dataset = Dataset(
            name=name,
            description=description,
            source_path=str(file_path),
        )
        session.add(dataset)
        await session.flush()

    for row in rows:
        call_log_id = str(row.get("callLogId") or row.get("call_id") or "").strip()
        if not call_log_id:
            continue
        transcript = {
            "callLogId": call_log_id,
            "number": row.get("number"),
            "messages": row.get("messages") or [],
            "updatedAt": row.get("updatedAt"),
        }
        session.add(
            CallRecord(
                dataset_id=dataset.id,
                call_log_id=call_log_id,
                number=row.get("number"),
                public_url=row.get("public_url"),
                human_url=row.get("human_url") or row.get("humanUrl"),
                recording_url=row.get("recordingUrl") or row.get("recording_url"),
                transcript_json=transcript,
            )
        )

    dataset.call_count = len(rows)
    await session.commit()
    await session.refresh(dataset)
    return dataset


async def get_dataset(session: AsyncSession, dataset_id: int) -> Dataset | None:
    return await session.get(Dataset, dataset_id)


async def list_datasets(session: AsyncSession) -> list[Dataset]:
    result = await session.scalars(select(Dataset).order_by(Dataset.created_at.desc()))
    return list(result.all())
