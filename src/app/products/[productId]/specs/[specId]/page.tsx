"use client";

import { useParams } from "next/navigation";
import { SpecEditor } from "@/components/SpecEditor";

export default function SpecPage() {
  const params = useParams<{ productId: string; specId: string }>();
  return <SpecEditor productId={params.productId} specId={params.specId} />;
}
