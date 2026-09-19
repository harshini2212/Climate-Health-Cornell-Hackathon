# Lane status

One file per lane, so six agents never conflict on a merge. Each agent appends two lines
when it finishes a task:

```
## 17:40 — cohort/build.py
10,000 veterans, ZIP weights from ACS B21001, all PLACES rates within 20% of source. make check green.
```

Read these with `make status`, not by opening them.
