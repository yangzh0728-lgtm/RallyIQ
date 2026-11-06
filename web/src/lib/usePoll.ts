import { useEffect, useRef, useState } from "react";

/** Poll `fn` every `ms` until `done(result)` is true. */
export function usePoll<T>(fn: (() => Promise<T>) | null, ms: number, done: (v: T) => boolean) {
  const [value, setValue] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const doneRef = useRef(done);
  doneRef.current = done;
  useEffect(() => {
    if (!fn) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const v = await fn();
        if (!alive) return;
        setValue(v);
        setError(null);
        if (!doneRef.current(v)) timer = setTimeout(tick, ms);
      } catch (e) {
        if (!alive) return;
        setError(String((e as Error).message ?? e));
        timer = setTimeout(tick, ms * 3);
      }
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [fn, ms]);
  return { value, error };
}
