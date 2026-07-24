"use client";

import { TopBar } from "@/components/ui";
import { Brand } from "@/components/Brand";
import ThemeToggle from "@/components/theme-toggle";
import LanguageToggle from "@/components/LanguageToggle";
import { NotificationTray } from "@/components/notifications/NotificationTray";
import { UserMenu } from "@/components/UserMenu";

/**
 * Shared TopBar chrome: Productarium brand (left) + language/theme toggles +
 * notification tray + user menu (right). Used across all pages so the header
 * stays consistent.
 */
export function AppHeader({ rightExtra }: { rightExtra?: React.ReactNode }) {
  return (
    <TopBar
      left={<Brand />}
      right={
        <>
          {rightExtra}
          <LanguageToggle />
          <ThemeToggle />
          <NotificationTray />
          <UserMenu />
        </>
      }
    />
  );
}

export default AppHeader;
