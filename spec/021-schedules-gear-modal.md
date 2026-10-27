# SPEC-021: Планировщик автогенерации — кнопка-шестерёнка с модалкой на странице продукта

Статус: planned · Волна 12 · Приоритет: средний

## Проблема

`ProductSchedulesPanel` рендерится отдельной секцией в теле страницы продукта (src/app/products/[productId]/page.tsx:1594-1603, только admin). Секция занимает место у всех администраторов; компактнее — кнопка-шестерёнка вверху продукта, открывающая модалку с тем же расписанием.

## Решение

- Убрать рендер секции из потока страницы (1594-1603).
- В хедере продукта рядом с карандашом редактирования (~786-795) для `user?.role === "admin"` — IconButton с иконкой Gear: aria-label/title = `t.schedulesTitle` («Автогенерация»); открывает `Modal size="xl"` c тем же заголовком, внутри — `<ProductSchedulesPanel productId entities embedded />`.
- `ProductSchedulesPanel`: новый проп `embedded?: boolean` — при true не рендерить собственный `<section>`+SectionHeader (заголовок уже в модалке), Card без mt-6; без пропа поведение прежнее (обратная совместимость не нужна других callers нет — проп можно сделать обязательным semantics минимального диффа).
- Логику загрузки/сохранения (getJson/putJson/deleteJson, usePanelAction) не трогать.

## Критерии приёмки

- Не-админ не видит шестерёнку; админ открывает модалку — внутри то же управление расписанием (toggle, пресеты/cron, delete, next/last run).
- Секция «Автогенерация» больше не рендерится в потоке страницы; `bun run lint` / `bun run build` зелёные.

## Затрагиваемые файлы

`src/app/products/[productId]/page.tsx`, `src/components/ProductSchedulesPanel.tsx`.
