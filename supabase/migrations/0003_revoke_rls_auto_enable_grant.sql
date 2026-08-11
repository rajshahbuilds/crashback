-- STU-46 — harden the pre-existing rls_auto_enable() event-trigger function.
-- It is SECURITY DEFINER and was executable by anon/authenticated via PostgREST RPC
-- (Supabase security lints 0028/0029). It is only meant to fire as a DDL event trigger,
-- so no API role should be able to call it. Guarded so it is a no-op where the function
-- does not exist (e.g. a fresh project built from these migrations alone).
do $$
begin
  if exists (
    select 1 from pg_proc p
    join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public' and p.proname = 'rls_auto_enable'
  ) then
    revoke execute on function public.rls_auto_enable() from anon, authenticated;
  end if;
end;
$$;
