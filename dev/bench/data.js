window.BENCHMARK_DATA = {
  "lastUpdate": 1794423736397,
  "repoUrl": "https://github.com/pattonw/tems",
  "entries": {
    "tems": [
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a52dad566241a67bb47d8f452db05ed7839cc538",
          "message": "Merge pull request #10 from pattonw/docs-deploy-on-tags\n\nDeploy docs on version tags and by hand only",
          "timestamp": "2026-11-11T19:01:01Z",
          "url": "https://github.com/pattonw/tems/commit/a52dad566241a67bb47d8f452db05ed7839cc538"
        },
        "date": 1794423736098,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import tems (cold)",
            "value": 1177.004,
            "range": "± 36.575",
            "unit": "ms",
            "extra": "interpreter startup 8.4 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 8.626,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      }
    ]
  }
}