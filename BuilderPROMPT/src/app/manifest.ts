import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "BillBox — Invoice & Bill Scanner",
    short_name: "BillBox",
    description: "Scan invoices and bills, extract the details automatically.",
    start_url: "/scan",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#4338ca",
    orientation: "portrait-primary",
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      {
        src: "/icons/icon-maskable-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
