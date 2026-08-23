# Private whole-rock database placeholder

Place the private SQLite database at:

```text
legacy_whole_rock_geochem/geochem.db
```

Element2Vec V2 expects the `v_high_quality_dataset` view. If the view is absent, inspect `research_views_documentation.md` and run `create_research_views.py` only after backing up the database and confirming the schema.

The database is intentionally not distributed and is ignored by Git.
