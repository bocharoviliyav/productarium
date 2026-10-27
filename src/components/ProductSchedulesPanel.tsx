"use client";

/**
 * «Автогенерация документации» (SPEC-009): per codebase/database cron
 * schedules, rendered for admins (product page modal, SPEC-021). Toggle + presets
 * (daily 02:00 / weekly Sun 03:00) or a custom cron. Server time is UTC;
 * a tick that finds the entity busy is skipped, not queued. Runs land in
 * the artifact version history (source "schedule").
 */

import { useCallback, useEffect, useState } from "react";
import { Trash } from "@phosphor-icons/react";
import { useLanguage } from "@/contexts/LanguageContext";
import { usePanelAction } from "@/hooks/usePanelAction";
import {
  Button,
  Card,
  IconButton,
  Input,
  Label,
  SectionHeader,
  Select,
  Spinner,
  Switch,
  Tag,
} from "@/components/ui";
import { ApiError, deleteJson, getJson, putJson } from "@/lib/api";
import type { DocgenSchedule, TagTone } from "@/lib/types";

type ScheduleEntity = { id: string; name: string; kind: "codebase" | "database" };

const PRESETS = { daily: "0 2 * * *", weekly: "0 3 * * 0" } as const;
type Preset = keyof typeof PRESETS | "custom";

/** `2026-10-26T02:00:00+00:00` → `2026-10-26 02:00 UTC`. */
function fmtUtc(iso: string | null | undefined): string | null {
  return iso ? `${iso.slice(0, 16).replace("T", " ")} UTC` : null;
}

function presetOf(cron: string): Preset {
  if (cron === PRESETS.daily) return "daily";
  if (cron === PRESETS.weekly) return "weekly";
  return "custom";
}

/** Label/tone per schedule.last_result (see api/docgen/scheduler.py). */
const RESULTS: Record<string, { key: string; tone: TagTone }> = {
  submitted: { key: "schedulesResultSubmitted", tone: "blue" },
  ok: { key: "schedulesResultOk", tone: "green" },
  skipped_busy: { key: "schedulesResultSkippedBusy", tone: "yellow" },
  error: { key: "schedulesResultError", tone: "red" },
  cancelled: { key: "schedulesResultCancelled", tone: "neutral" },
};

function ScheduleRow({
  productId,
  entity,
  schedule,
  onReload,
}: {
  productId: string;
  entity: ScheduleEntity;
  schedule: DocgenSchedule | undefined;
  onReload: () => Promise<void>;
}) {
  const { messages } = useLanguage();
  const t = messages?.product ?? {};
  const { busy, run } = usePanelAction(t.schedulesTitle ?? "Automatic documentation");
  const [cron, setCron] = useState(schedule?.cron_expr ?? PRESETS.daily);

  // Re-sync the draft when the persisted schedule changes (save/toggle).
  useEffect(() => {
    setCron(schedule?.cron_expr ?? PRESETS.daily);
  }, [schedule?.cron_expr]);

  const save = (next: { cron: string; enabled: boolean }) =>
    run(async () => {
      await putJson(`/api/products/${productId}/schedules`, {
        entity_type: entity.kind,
        entity_id: entity.id,
        cron_expr: next.cron.trim(),
        enabled: next.enabled,
      });
      await onReload();
    });

  const remove = () =>
    run(async () => {
      await deleteJson(
        `/api/products/${productId}/schedules?entity_type=${entity.kind}&entity_id=${encodeURIComponent(entity.id)}`,
      );
      await onReload();
    });

  const enabled = schedule?.enabled ?? false;
  const stored = schedule?.cron_expr ?? PRESETS.daily;
  const result = schedule?.last_result ? (RESULTS[schedule.last_result] ?? null) : null;
  const nextRun = fmtUtc(schedule?.next_run_at);
  const lastRun = fmtUtc(schedule?.last_run_at);
  const kindLabel =
    entity.kind === "codebase"
      ? (messages?.artifactTypes?.codebase?.label ?? "Codebase")
      : (messages?.artifactTypes?.database?.label ?? "Database");

  return (
    <div className="p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2.5">
          <Switch
            checked={enabled}
            onChange={(next) => void save({ cron, enabled: next })}
            disabled={busy}
            label={t.schedulesToggleLabel ?? "Enable or disable automatic generation"}
          />
          <span className="truncate text-sm font-medium text-ink">{entity.name}</span>
          <Tag tone={entity.kind === "codebase" ? "blue" : "green"}>{kindLabel}</Tag>
          {!enabled && <Tag tone="neutral">{t.schedulesDisabled ?? "off"}</Tag>}
        </div>
        {schedule && (
          <IconButton
            type="button"
            aria-label={t.schedulesDelete ?? "Delete schedule"}
            title={t.schedulesDelete ?? "Delete schedule"}
            onClick={() => void remove()}
            disabled={busy}
          >
            {busy ? <Spinner /> : <Trash size={14} weight="regular" />}
          </IconButton>
        )}
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,220px)_minmax(0,1fr)_auto]">
        <div>
          <Label>{t.schedulesPreset ?? "Schedule"}</Label>
          <Select
            value={presetOf(cron)}
            onChange={(e) => {
              const preset = e.target.value as Preset;
              if (preset !== "custom") setCron(PRESETS[preset]);
            }}
            disabled={busy}
          >
            <option value="daily">{t.schedulesPresetDaily ?? "Daily at 02:00"}</option>
            <option value="weekly">{t.schedulesPresetWeekly ?? "Weekly, Sun 03:00"}</option>
            <option value="custom">{t.schedulesPresetCustom ?? "Custom cron"}</option>
          </Select>
        </div>
        <div>
          <Label>{t.schedulesCron ?? "Cron (UTC)"}</Label>
          <Input
            value={cron}
            onChange={(e) => setCron(e.target.value)}
            className="font-mono"
            spellCheck={false}
            autoComplete="off"
            disabled={busy}
          />
        </div>
        <div className="flex items-end">
          <Button
            size="sm"
            disabled={busy || !cron.trim() || cron.trim() === stored}
            onClick={() => void save({ cron, enabled })}
          >
            {busy ? <Spinner /> : null}
            {t.schedulesSave ?? "Save"}
          </Button>
        </div>
      </div>

      {(nextRun || lastRun || schedule?.last_result) && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
          {nextRun && (
            <span>
              {t.schedulesNextRun ?? "Next run"}: <span className="font-mono">{nextRun}</span>
            </span>
          )}
          {lastRun && (
            <span>
              {t.schedulesLastRun ?? "Last run"}: <span className="font-mono">{lastRun}</span>
            </span>
          )}
          {schedule?.last_result && (
            <Tag tone={result?.tone ?? "neutral"}>
              {result ? (t[result.key] ?? schedule.last_result) : schedule.last_result}
            </Tag>
          )}
        </div>
      )}
    </div>
  );
}

export function ProductSchedulesPanel({
  productId,
  entities,
}: {
  productId: string;
  entities: ScheduleEntity[];
}) {
  const { messages } = useLanguage();
  const t = messages?.product ?? {};
  const { fail } = usePanelAction(t.schedulesTitle ?? "Automatic documentation");
  const [schedules, setSchedules] = useState<DocgenSchedule[] | null>(null);

  const load = useCallback(async () => {
    try {
      setSchedules(await getJson<DocgenSchedule[]>(`/api/products/${productId}/schedules`));
    } catch (e) {
      setSchedules([]);
      if (e instanceof ApiError && e.status === 403) return; // lost settings rights
      fail(e);
    }
  }, [productId, fail]);

  useEffect(() => {
    if (entities.length === 0) return;
    void load();
  }, [load, entities.length]);

  if (entities.length === 0) return null;

  const byEntity = new Map(
    (schedules ?? []).map((s) => [`${s.entity_type}:${s.entity_id}`, s] as const),
  );

  const body = schedules === null ? (
    <div className="flex items-center gap-2 p-4 text-sm text-muted">
      <Spinner /> {t.schedulesLoading ?? "Loading…"}
    </div>
  ) : (
    <div className="divide-y divide-divider">
      {entities.map((entity) => (
        <ScheduleRow
          key={entity.id}
          productId={productId}
          entity={entity}
          schedule={byEntity.get(`${entity.kind}:${entity.id}`)}
          onReload={load}
        />
      ))}
    </div>
  );

  return (
    <section>
      <SectionHeader
        title={t.schedulesTitle ?? "Automatic documentation"}
        subtitle={t.schedulesSubtitle ?? ""}
      />
      <Card className="mt-6 overflow-hidden">{body}</Card>
    </section>
  );
}
