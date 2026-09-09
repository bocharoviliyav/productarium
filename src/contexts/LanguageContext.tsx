/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import enMessages from "../messages/en.json";

type Messages = Record<string, any>;
/** Interpolate `{name}` placeholders in a template string. */
export function fmt(
  template: string | undefined,
  vars?: Record<string, string | number>,
): string {
  if (!template) return "";
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_, k: string) =>
    k in vars ? String(vars[k]) : `{${k}}`,
  );
}
type LanguageContextType = {
  language: string;
  setLanguage: (lang: string) => void;
  messages: Messages;
  supportedLanguages: Record<string, string>;
  /** Interpolate `{name}` placeholders in a message template. */
  fmt: (template: string | undefined, vars?: Record<string, string | number>) => string;
};

const LanguageContext = createContext<LanguageContextType | undefined>(
  undefined,
);

export function LanguageProvider({ children }: { children: ReactNode }) {
  // Initialize with 'en' or get from localStorage if available
  const [language, setLanguageState] = useState<string>("en");
  // P2-31: the app renders IMMEDIATELY with the bundled English messages;
  // the user's language (localStorage / browser) is applied asynchronously
  // once /api/lang/config resolves — no full-app spinner gate.
  const [messages, setMessages] = useState<Messages>(enMessages);
  const [supportedLanguages, setSupportedLanguages] = useState({});
  const [defaultLanguage, setDefaultLanguage] = useState("en");

  // Helper function to detect browser language
  const detectBrowserLanguage = (): string => {
    try {
      if (typeof window === "undefined" || typeof navigator === "undefined") {
        return "en"; // Default to English on server-side
      }

      // Get browser language (navigator.language returns full locale like 'en-US')
      const browserLang =
        navigator.language || (navigator as any).userLanguage || "";
      if (!browserLang) {
        return "en";
      }

      // Extract the language code (first 2 characters)
      const langCode = browserLang.split("-")[0].toLowerCase();

      // Check if the detected language is supported: the fetched config
      // when available, else the message files known to ship with the app
      // (used before /api/lang/config resolves).
      const supported = Object.keys(supportedLanguages);
      const known = supported.length > 0 ? supported : ["en", "ru"];
      return known.includes(langCode) ? langCode : "en";
    } catch {
      return "en";
    }
  };

  useEffect(() => {
    const getSupportedLanguages = async () => {
      try {
        const response = await fetch("/api/lang/config");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setSupportedLanguages(data.supported_languages);
        setDefaultLanguage(data.default);
      } catch {
        // Offline / backend down: fall back to the bundled set — the app is
        // already rendered with English messages, nothing blocks.
        const defaultSupportedLanguages = {
          en: "English",
          ru: "Русский (Russian)",
        };
        setSupportedLanguages(defaultSupportedLanguages);
        setDefaultLanguage("en");
      }
    };
    getSupportedLanguages();
  }, []);

  useEffect(() => {
    if (Object.keys(supportedLanguages).length > 0) {
      const loadLanguage = async () => {
        try {
          // Only access localStorage in the browser
          let storedLanguage;
          if (typeof window !== "undefined") {
            storedLanguage = localStorage.getItem("language");

            // If no language is stored, detect browser language
            if (!storedLanguage) {
              storedLanguage = detectBrowserLanguage();
              localStorage.setItem("language", storedLanguage);
            }
          } else {
            storedLanguage = "en";
          }

          const validLanguage = Object.keys(supportedLanguages).includes(
            storedLanguage as any,
          )
            ? storedLanguage
            : defaultLanguage;

          // Load messages for the language and swap them in — the app stays
          // interactive (English) until this resolves.
          const langMessages = (
            await import(`../messages/${validLanguage}.json`)
          ).default;

          setLanguageState(validLanguage);
          setMessages(langMessages);

          // Update HTML lang attribute (only in browser)
          if (typeof document !== "undefined") {
            document.documentElement.lang = validLanguage;
          }
        } catch (error) {
          console.error("Failed to load language:", error);
          // Fallback to English (already rendered — nothing to un-block)
          try {
            const fallback = (await import("../messages/en.json")).default;
            setMessages(fallback);
          } catch {
            // keep the statically bundled English messages in state
          }
        }
      };

      loadLanguage();
    }
  }, [supportedLanguages, defaultLanguage]);

  // Update language and load new messages
  const setLanguage = async (lang: string) => {
    try {
      const validLanguage = Object.keys(supportedLanguages).includes(
        lang as any,
      )
        ? lang
        : defaultLanguage;

      // Load messages for the new language
      const langMessages = (await import(`../messages/${validLanguage}.json`))
        .default;

      setLanguageState(validLanguage);
      setMessages(langMessages);

      // Store in localStorage (only in browser)
      if (typeof window !== "undefined") {
        localStorage.setItem("language", validLanguage);
      }

      // Update HTML lang attribute (only in browser)
      if (typeof document !== "undefined") {
        document.documentElement.lang = validLanguage;
      }
    } catch (error) {
      console.error("Failed to set language:", error);
    }
  };

  return (
    <LanguageContext.Provider
      value={{ language, setLanguage, messages, supportedLanguages, fmt }}
    >
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
