create table if not exists cache_details (
       id integer primary key,
       cache_id integer not null,
       fn_name text not null,
       opt_name text,
       opt_value text,
       opt_type text
       );
