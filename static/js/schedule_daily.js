// static/js/schedule_daily.js

document.addEventListener("DOMContentLoaded", () => {
  const monthInput = document.getElementById("month-input");
  const loadBtn = document.getElementById("load-btn");
  const tableBody = document.querySelector("#daily-table tbody");
  const statusMsg = document.getElementById("status-msg");

  // 현재 월 설정
  const now = new Date();
  monthInput.value = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}`;

  let currentScheduleId = null;

  async function fetchJSON(url, opts={}) {
    const res = await fetch(url, {...opts, headers: {'Content-Type': 'application/json'}});
    if(!res.ok) throw new Error(await res.text());
    return res.json();
  }

  function getDayName(dateStr) {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return days[new Date(dateStr).getDay()];
  }

  async function loadData() {
    const month = monthInput.value;
    if(!month) return;

    statusMsg.textContent = "데이터 로드 중...";
    tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">로딩 중...</td></tr>`;

    try {
      // 1. 해당 월의 스케줄 ID 조회
      const meta = await fetchJSON(`/api/schedules/by_month?month=${month}`);
      if(!meta.exists) {
        tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">해당 월의 스케줄이 없습니다. 먼저 스케줄을 생성해주세요.</td></tr>`;
        statusMsg.textContent = "";
        return;
      }
      currentScheduleId = meta.schedule_id;

      // 2. 일별 데이터 조회
      const dailyData = await fetchJSON(`/api/schedules/${currentScheduleId}/daily`);
      renderTable(dailyData);
      statusMsg.textContent = "조회 완료";

    } catch(e) {
      console.error(e);
      tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:red;">오류 발생</td></tr>`;
      statusMsg.textContent = "오류 발생: " + e.message;
    }
  }

  function renderTable(data) {
    tableBody.innerHTML = "";

    if(!data || data.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">표시할 데이터가 없습니다.</td></tr>`;
        return;
    }

    data.forEach(row => {
      const tr = document.createElement("tr");

      // 날짜
      const dayName = getDayName(row.date);
      const dateParts = row.date.split("-"); // [YYYY, MM, DD]
      tr.innerHTML = `
        <td class="date-col weekday-${dayName}">
          ${dateParts[2]}일<br>
          <small>(${dayName})</small>
        </td>
      `;

      // Shifts (D, E, N)
      ["D", "E", "N"].forEach(shift => {
        const td = document.createElement("td");
        const members = row.shifts[shift] || [];

        // 리더를 맨 앞으로, 그리고 태그 생성
        const html = members.map(m => {
          const leaderClass = m.is_leader ? "leader" : "";
          const icon = m.is_leader ? "<span class='leader-badge'>★</span>" : "";
          return `<span class="member-tag ${leaderClass}">${icon}${m.name}</span>`;
        }).join("");

        td.innerHTML = html || "<span style='color:#ccc;'>-</span>";
        tr.appendChild(td);
      });

      // Note (특이사항)
      const tdNote = document.createElement("td");
      const textarea = document.createElement("textarea");
      textarea.className = "note-textarea";
      textarea.value = row.note || "";
      textarea.placeholder = "메모 입력...";

      // 메모 변경 시 자동 저장 (Debounce)
      let timer;
      textarea.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(() => saveNote(row.date, textarea.value), 1000);
      });

      tdNote.appendChild(textarea);
      tr.appendChild(tdNote);

      tableBody.appendChild(tr);
    });
  }

  async function saveNote(date, note) {
    if(!currentScheduleId) return;
    try {
      await fetchJSON(`/api/schedules/${currentScheduleId}/daily/note`, {
        method: "POST",
        body: JSON.stringify({ date, note })
      });
      console.log(`Note saved for ${date}`);
    } catch(e) {
      console.error("Note save failed", e);
      alert("메모 저장 실패");
    }
  }

  // 인쇄 기능
  document.getElementById("print-btn").addEventListener("click", () => {
    window.print();
  });

  loadBtn.addEventListener("click", loadData);

  // 초기 로드
  loadData();
});