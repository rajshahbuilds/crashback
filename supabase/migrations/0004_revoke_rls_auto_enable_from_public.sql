-- STU-46 — actually remove anon/authenticated EXECUTE on rls_auto_enable().
-- 0003 revoked from anon/authenticated directly, but the function's EXECUTE is granted to
-- PUBLIC (Postgres default), which those roles inherit — so the grant survived. Revoke from
-- PUBLIC as well; postgres and service_role keep their explicit grants. Guarded for projects
-- without the function.
do $$
begin
  if exists (
    select 1 from pg_proc p
    join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public' and p.proname = 'rls_auto_enable'
  ) then
    revoke execute on function public.rls_auto_enable() from public, anon, authenticated;
  end if;
end;
$$;
