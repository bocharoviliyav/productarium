import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { ThemeProvider } from "next-themes";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { AuthProvider } from "@/contexts/AuthContext";
import { NotificationProvider } from "@/contexts/NotificationContext";
import { ToastViewport } from "@/components/notifications/ToastViewport";

export const metadata: Metadata = {
  title: "Productarium — Product Knowledge",
  description:
    "Generate, curate, and verify documentation for your products and their artifacts.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${GeistSans.variable} ${GeistMono.variable} antialiased`}
      >
        <ThemeProvider attribute="data-theme" defaultTheme="light" enableSystem>
          <LanguageProvider>
            <AuthProvider>
              <NotificationProvider>
                {children}
                <ToastViewport />
              </NotificationProvider>
            </AuthProvider>
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
