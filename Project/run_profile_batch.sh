#!/usr/bin/env bash
# run_profile_batch.sh — 7 videoyu PROFILE=1 ile sırayla çalıştırır
# Her video için ayrı profile_<stem>.jsonl üretir
# Kullanım: bash run_profile_batch.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="C:/Users/cagataykonuralp/AppData/Local/Programs/Python/Python310/python.exe"
VIDEOS=(
  "V:/web_client_CAG_1939-0003-1-000-00-1-KÜÇÜK_KIZIN_RÜYASI.mp4"
  "V:/web_client_CAG_1965-0057-1-0000-00-1-NEW_YORK'TAKİ_JANDARMA.mp4"
  "V:/web_client_CAG_1973-0177-1-0000-90-1-AĞLIYORUM.mp4"
  "V:/web_client_CAG_1989-0058-1-0000-00-1-ESAS_HEDEF.mp4"
  "V:/web_client_CAG_1995-0288-1-0000-00-1-KARA_RAHİP.mp4"
  "V:/web_client_CAG_2011-9252-0-0001-88-1-DİLEKLER_ZAMANI.mp4"
  "V:/web_client_CAG_2021-2164-1-0000-56-1-BEYAZ_BALON.mp4"
)

TOTAL=${#VIDEOS[@]}
PASS=0
FAIL=0

for i in "${!VIDEOS[@]}"; do
  VIDEO="${VIDEOS[$i]}"
  # stem = dosya adından yol ve uzantı çıkarıldı
  STEM=$(basename "$VIDEO" .mp4)
  LOG_FILE="$SCRIPT_DIR/profile_${STEM}.jsonl"
  NUM=$((i+1))

  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  [$NUM/$TOTAL] $STEM"
  echo "  Log → $LOG_FILE"
  echo "════════════════════════════════════════════════════════"

  PROFILE=1 \
  PROFILE_LOG="$LOG_FILE" \
  HEADLESS=1 \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    "$PYTHON" main.py "$VIDEO"

  EXIT=$?
  if [ $EXIT -eq 0 ]; then
    PASS=$((PASS+1))
    echo "  ✓ Tamamlandı (exit 0)"
  else
    FAIL=$((FAIL+1))
    echo "  ✗ HATA (exit $EXIT)"
  fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ÖZET: $PASS/$TOTAL başarılı, $FAIL hata"
echo "════════════════════════════════════════════════════════"

# Özet tablosunu yazdır
echo ""
echo "Profil logları:"
for VIDEO in "${VIDEOS[@]}"; do
  STEM=$(basename "$VIDEO" .mp4)
  LOG_FILE="$SCRIPT_DIR/profile_${STEM}.jsonl"
  if [ -f "$LOG_FILE" ]; then
    LINES=$(wc -l < "$LOG_FILE")
    echo "  $STEM → $LINES satır"
  else
    echo "  $STEM → LOG YOK"
  fi
done
