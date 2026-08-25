# InferenceX — Known Issues & Improvement Backlog

## [ISSUE-1] _work directory bloccata da file root-owned su NFS (mia1-p01-g07)

**Frequenza:** Alta — si verifica ad ogni run fallito/cancellato prima del chown di cleanup.

**Causa:**
Il container Docker gira come root. Su NFS con `root_squash`, il root del container viene mappato a `nobody` (uid 65534). Al termine del job il launcher tenta `sudo -n chown` per riconsegnare i file, ma su `mia1-p01-g07` l'utente non ha sudo → i file in `_work/` restano owned da `nobody` → il runner non riesce a cancellare la working directory al run successivo (EACCES su `git clean` / `actions/checkout`).

**Workaround attuale (manuale):**
Lanciare un container Docker con `--user root` per acquisire i permessi e cancellare la directory:

```bash
docker run --rm \
  -v /it-share/gguasti/actions-runner/_work:/work \
  ubuntu:22.04 \
  rm -rf /work/InferenceX
```

**Fix desiderati:**

1. **Configurare sudo passwordless per chown** — aggiungere in `/etc/sudoers.d/gguasti` sulla macchina:
   ```
   giovanni.guasti@amd.com ALL=(ALL) NOPASSWD: /usr/bin/chown
   ```
   Richiede accesso admin alla macchina.

2. **Usare `--user $(id -u):$(id -g)` nel docker run** — evita che i file vengano creati come root. Richiede modifica al launcher e verifica che lo script benchmark funzioni senza root.

3. **Pre-cleanup via Docker nel launcher** — prima di `git checkout`, il launcher potrebbe auto-pulire la `_work` con un container root. Più complesso ma completamente automatico.

---

## [ISSUE-2] salloc presente ma partizione SLURM non configurata (mia1-p01-g07)

**Causa:** La macchina ha SLURM installato come client ma non ha la partizione `compute`. Il Docker fallback nel launcher scatta solo se `salloc` è assente, quindi SLURM viene usato e fallisce.

**Fix applicato (branch testgg):** Aggiunto `FORCE_DOCKER=1` nel `.env` del runner + controllo nel launcher:
```bash
if ! command -v salloc >/dev/null 2>&1 || [[ "${FORCE_DOCKER:-}" == "1" ]]; then
```

**Stato:** Risolto. Runner `.env` su mia1-p01-g07 ha `FORCE_DOCKER=1`.

---

## [ISSUE-3] sudo blocca in attesa password (mia1-p01-g07)

**Causa:** `sudo chown` senza `-n` aspettava input da stdin (3 tentativi × ~60s) bloccando il runner.

**Fix applicato (branch testgg):** Tutti i `sudo` nel Docker fallback ora usano `sudo -n` → fallisce immediatamente senza password, `|| true` gestisce l'errore.

**Stato:** Risolto.
