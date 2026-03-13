*** Remove containers ***
```bash
$ docker compose down
```

*** Remove logs ***
```bash
$ rm -rf core-data/ core-logs/
```

*** Create needed containers ***
```bash
$ docker compose up --build -d
```

*** Run the app ***
```bash
$ python -m task.app 
```
