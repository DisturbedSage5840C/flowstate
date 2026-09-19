import { useSyncExternalStore } from 'react'

export type Theme = 'light' | 'dark'
const KEY = 'flowstate.theme'
const listeners = new Set<() => void>()

const stored = (): Theme | null => {
  try {
    const s = localStorage.getItem(KEY)
    return s === 'light' || s === 'dark' ? s : null
  } catch {
    return null
  }
}
const system = (): Theme => (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')

const apply = (t: Theme) => {
  document.documentElement.dataset.theme = t
  listeners.forEach((l) => l())
}

/* Call once at startup (index.html also sets it pre-paint). An explicit choice wins; otherwise follow the OS live. */
export function initTheme() {
  apply(stored() ?? system())
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!stored()) apply(system())
  })
}

export function setTheme(t: Theme) {
  try {
    localStorage.setItem(KEY, t)
  } catch {
    /* per-viewer convenience only */
  }
  apply(t)
}

const read = (): Theme => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')
const subscribe = (l: () => void) => {
  listeners.add(l)
  return () => listeners.delete(l)
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, read)
  return { theme, dark: theme === 'dark', toggle: () => setTheme(theme === 'dark' ? 'light' : 'dark') }
}
