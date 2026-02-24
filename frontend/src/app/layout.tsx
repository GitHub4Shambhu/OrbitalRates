import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OrbitalRates — Fixed Income Relative Value",
  description: "AI-Native Global Fixed Income Relative Value Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
