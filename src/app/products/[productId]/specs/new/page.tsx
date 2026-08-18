"use client";

import { useParams } from "next/navigation";
import { SpecEditor } from "@/components/SpecEditor";

export default function NewSpecPage() {
  const params = useParams<{ productId: string }>();
  return <SpecEditor productId={params.productId} specId={null} />;
}
