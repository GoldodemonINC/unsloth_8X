// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

/** `owner/name`, the only shape the Hub resolves as a repository. */
export const REPO_ID_RE =
  /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;

const HF_URL_RE =
  /^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(?:(?:datasets|models)\/)?([^?#\s]+)/i;

const GIT_SUFFIX_RE = /\.git$/i;

/**
 * Reads a Hub reference the way people paste it: a bare `owner/name`, or a page
 * URL with its `/datasets`, `/models` and `/tree/<rev>` parts. Returns null for
 * anything that is not a repository, so the caller can say why instead of
 * storing text that will never resolve.
 *
 * A file page (`/blob/…`, `/resolve/…`) is refused rather than widened to its
 * owner: the field configures a repository, and silently dropping the file part
 * of what someone pasted would misstate what the run will use.
 */
export function parseRepoId(raw: string): string | null {
  const text = raw.trim();
  if (!text) {
    return null;
  }
  const withoutGit = text.replace(GIT_SUFFIX_RE, "");
  const fromUrl = HF_URL_RE.exec(withoutGit);
  if (!fromUrl) {
    return REPO_ID_RE.test(withoutGit) ? withoutGit : null;
  }
  const [owner, name, ...rest] = fromUrl[1].split("/");
  if (!(owner && name)) {
    return null;
  }
  if (rest.some((part) => part === "blob" || part === "resolve")) {
    return null;
  }
  const repoId = `${owner}/${name.replace(GIT_SUFFIX_RE, "")}`;
  return REPO_ID_RE.test(repoId) ? repoId : null;
}
