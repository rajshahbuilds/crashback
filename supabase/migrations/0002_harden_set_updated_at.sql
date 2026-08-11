-- STU-46 — harden the set_updated_at trigger function.
-- Pin an empty search_path so the function cannot be hijacked via a mutable search_path
-- (Supabase security lint 0011). now() resolves from pg_catalog regardless.
create or replace function set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;
