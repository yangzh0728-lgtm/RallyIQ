// Video files the backend accepts: same list as rallyiq/api/app.py ALLOWED_EXT.
export const VIDEO_EXT = [".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"];

/** Accepted by extension; a file without an extension is accepted when the browser says it is a video. */
export const isVideoFile = (f: { name: string; type: string }) =>
  VIDEO_EXT.some((e) => f.name.toLowerCase().endsWith(e)) || (f.type.startsWith("video/") && !/\.[a-z0-9]+$/i.test(f.name));
