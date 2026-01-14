// static/js/personnel.js

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const nurseForm = document.getElementById("nurse-form");
  const nurseIdInput = document.getElementById("nurse-id");
  const staffIdInput = document.getElementById("staff-id");
  const staffDupMsg = document.getElementById("staff-dup-msg");
  const nameInput = document.getElementById("name");
  const joinYearInput = document.getElementById("join-year");
  const yearsExpInput = document.getElementById("years-exp");

  const levelSelect = document.getElementById("level-label");
  const levelCustomInput = document.getElementById("level-custom");

  const leaderDInput = document.getElementById("leader-d");
  const leaderEInput = document.getElementById("leader-e");
  const leaderNInput = document.getElementById("leader-n");
  const relWorkInput = document.getElementById("relative-work-days");

  // Checkboxes
  const tagCheckboxes = document.querySelectorAll('input[name="nurse-tag"]');
  const prefCheckboxes = document.querySelectorAll('input[name="pref-shift"]');
  const avoidCheckboxes = document.querySelectorAll('input[name="avoid-shift"]');

  const teamInput = document.getElementById("team-input");
  const teamDatalist = document.getElementById("team-options");
  const preceptPartnerInput = document.getElementById("precept-partner");
  const preceptPartnerOptions = document.getElementById("precept-partner-options");
  const avoidInput = document.getElementById("avoid-input");
  const avoidOptions = document.getElementById("avoid-options");

  const detailSection = document.getElementById("detail-section");
  const toggleDetailBtn = document.getElementById("toggle-detail-btn");
  const formMessage = document.getElementById("form-message");
  const resetFormBtn = document.getElementById("reset-form-btn");
  const deleteNurseBtn = document.getElementById("delete-nurse-btn");
  const saveNurseBtn = document.getElementById("save-nurse-btn");

  const excelFileInput = document.getElementById("excel-file");
  const uploadExcelBtn = document.getElementById("upload-excel-btn");
  const uploadResult = document.getElementById("upload-result");
  const downloadNursesBtn = document.getElementById("download-nurses-btn");

  const nurseTableBody = document.querySelector("#nurse-table tbody");
  const nurseTableHead = document.querySelector("#nurse-table thead");
  const wardNameDisplay = document.getElementById("ward-name-display");

  // Global State
  let nursesCache = [];
  let knownTeams = new Set();
  let selectedNurseId = null;
  let sortState = { col: "name", dir: "asc" };
  const currentYear = new Date().getFullYear();

  // Initialize
  joinYearInput.value = currentYear;
  yearsExpInput.value = "0";
  if (deleteNurseBtn) deleteNurseBtn.style.display = "none";

  // Level Options
  levelSelect.innerHTML = "";
  [
    { value: "1.0", text: "일반 (3~5년차)" },
    { value: "0.7", text: "신입 (1~2년차)" },
    { value: "1.1", text: "리더 (6~10년차)" },
    { value: "1.2", text: "고연차리더 (10년차 이상)" },
    { value: "manual", text: "수동조정" }
  ].forEach(opt => {
    const el = document.createElement("option");
    el.value = opt.value; el.textContent = opt.text;
    levelSelect.appendChild(el);
  });

  // Toggle Detail
  if (toggleDetailBtn && detailSection) {
    toggleDetailBtn.addEventListener("click", () => {
      const isHidden = detailSection.classList.contains("hidden") || detailSection.style.display === "none";
      if (isHidden) {
        detailSection.style.display = "block";
        detailSection.classList.remove("hidden");
        toggleDetailBtn.textContent = "세부 설정 숨기기";
      } else {
        detailSection.style.display = "none";
        detailSection.classList.add("hidden");
        toggleDetailBtn.textContent = "세부 설정 열기";
      }
    });
  }

  // -----------------------------------------------------------
  // [New] UI Logic: Tags Interaction (Grey out & Toggle)
  // -----------------------------------------------------------
  function updateTagInteractions() {
    let hasAnyTag = false;
    let isPregnant = false;

    tagCheckboxes.forEach(cb => {
      if (cb.checked) {
        hasAnyTag = true;
        if (cb.value === 'pregnant') isPregnant = true;
      }
    });

    // 1. 특수 상태 선택 시 근무 선호도 Grey out (선택 불가)
    prefCheckboxes.forEach(cb => {
      cb.disabled = hasAnyTag;
      if (hasAnyTag) cb.checked = false; // 선택 시 기존 체크 해제
    });
    avoidCheckboxes.forEach(cb => {
      cb.disabled = hasAnyTag;
      if (hasAnyTag) cb.checked = false;
    });
  }

  // 이벤트 리스너 등록
  tagCheckboxes.forEach(cb => {
    cb.addEventListener('change', (e) => {
      updateTagInteractions();
      if (cb.value === 'pregnant') {
        if (cb.checked) {
          relWorkInput.value = "0.8";
        } else {
          if (relWorkInput.value === "0.8") {
            relWorkInput.value = "1.0";
          }
        }
      }
    });
  });

  // -----------------------------------------------------------
  // Helper Functions
  // -----------------------------------------------------------
  function showFormMessage(msg, isError = false) {
    if (!formMessage) return;
    formMessage.textContent = msg || "";
    formMessage.classList.toggle("error", isError);
    if (msg) setTimeout(() => { formMessage.textContent = ""; formMessage.classList.remove("error"); }, 4000);
  }
  function showUploadMessage(msg, isError = false) {
    if (!uploadResult) return;
    uploadResult.textContent = msg || "";
    uploadResult.classList.toggle("error", isError);
    if (msg) setTimeout(() => { uploadResult.textContent = ""; uploadResult.classList.remove("error"); }, 6000);
  }
  function setStaffDupState(isDup, message) {
    staffIdInput.classList.toggle("input-error", isDup);
    if (staffDupMsg) { staffDupMsg.textContent = message || ""; staffDupMsg.classList.toggle("error", isDup); }
  }
  function yearFromDateStr(s) {
    if (!s || typeof s !== "string") return null;
    const m = s.match(/^(\d{4})/); return m ? parseInt(m[1], 10) : null;
  }
  function extractTeamString(raw) {
    if (!raw) return "";
    if (typeof raw.team === "string") return raw.team;
    return raw.team_code || raw.team_name || "";
  }
  function stripIdFromLabel(label) {
    if (!label) return "";
    return label.split(",").map(p => p.split("(")[0].trim()).filter(Boolean).join(", ");
  }

  // -----------------------------------------------------------
  // Data Normalization (Fix for Leader)
  // -----------------------------------------------------------
  function normalizeNurse(raw) {
    const id = raw.nurse_id ?? raw.id;
    const staff = (raw.staff_id ?? raw.employee_no ?? "").toString().trim();
    const joinYear = raw.join_year ?? yearFromDateStr(raw.join_date);
    const yearsExp = raw.years_exp && raw.years_exp > 0 ? raw.years_exp
        : (typeof joinYear === "number" ? Math.max(1, currentYear - joinYear + 1) : 0);
    const level = raw.level ?? raw.level_weight ?? 1.0;
    const rel = raw.relative_work_days ?? raw.short_work_factor ?? 1.0;

    // [Fix] 리더 정보: leader_eligible 객체 우선 확인
    const le = raw.leader_eligible || {};
    // 개별 필드(leader_d)가 있으면 그것도 확인 (OR 조건)
    const leader_d = (le.D === true) || (raw.leader_d === true);
    const leader_e = (le.E === true) || (raw.leader_e === true);
    const leader_n = (le.N === true) || (raw.leader_n === true);

    return {
      nurse_id: id, staff_id: staff, name: raw.name || "",
      join_year: joinYear, years_exp: yearsExp,
      level: level, relative_work_days: rel,
      leader_d: leader_d,
      leader_e: leader_e,
      leader_n: leader_n,
      team: extractTeamString(raw),
      precept_partner: raw.precept_partner ?? "",
      avoid_list: raw.avoid_list ?? "",
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      preferred_shifts: Array.isArray(raw.preferred_shifts) ? raw.preferred_shifts : [],
      avoid_shifts: Array.isArray(raw.avoid_shifts) ? raw.avoid_shifts : [],
    };
  }

  function findByName(name) {
    const t = (name || "").trim(); if(!t) return null;
    return nursesCache.find(n => (n.name||"").trim() === t) || null;
  }

  // -----------------------------------------------------------
  // Logic: Level & Year Auto-Calc
  // -----------------------------------------------------------
  function updateLevelCustomVisibility() {
    const val = levelSelect.value;
    levelCustomInput.classList.remove("hidden");
    if (val === "manual") {
      levelCustomInput.disabled = false;
      if (levelCustomInput.value === "") levelCustomInput.value = "1.0";
    } else {
      levelCustomInput.disabled = true;
      const num = parseFloat(val);
      if (!isNaN(num)) levelCustomInput.value = num.toFixed(1);
    }
  }

  function autoCalculateLevel() {
    const exp = parseInt(yearsExpInput.value, 10)||0;
    const partnerName = preceptPartnerInput.value.trim();

    if (partnerName) {
      const partner = findByName(partnerName);
      if (partner && exp <= (partner.years_exp || 0)) {
        levelSelect.value = "manual";
        updateLevelCustomVisibility();
        levelCustomInput.value = "0.0";
        return;
      }
    }
    if (!selectedNurseId || levelSelect.value !== "manual" || parseFloat(levelCustomInput.value) === 0.0) {
      let targetVal = "1.0";
      if (exp <= 2) targetVal = "0.7";
      else if (exp <= 5) targetVal = "1.0";
      else if (exp <= 10) targetVal = "1.1";
      else targetVal = "1.2";
      levelSelect.value = targetVal;
      updateLevelCustomVisibility();
    }
  }

  joinYearInput.addEventListener("change", () => {
    const jy = parseInt(joinYearInput.value, 10);
    if (!isNaN(jy)) {
      const exp = currentYear - jy + 1;
      yearsExpInput.value = String(exp >= 1 ? exp : 1);
      autoCalculateLevel();
    }
  });
  yearsExpInput.addEventListener("change", () => {
    const exp = parseInt(yearsExpInput.value, 10);
    if (!isNaN(exp) && exp > 0) {
      const jy = currentYear - exp + 1;
      joinYearInput.value = String(jy);
      autoCalculateLevel();
    }
  });
  preceptPartnerInput.addEventListener("blur", autoCalculateLevel);
  levelSelect.addEventListener("change", updateLevelCustomVisibility);
  updateLevelCustomVisibility();

  // -----------------------------------------------------------
  // UI: Autocomplete (Team, Name)
  // -----------------------------------------------------------
  function fillTeamOptions() {
    knownTeams = new Set(); nursesCache.forEach(n => { if(n.team) knownTeams.add(n.team); });
    if(teamDatalist) {
      teamDatalist.innerHTML = "";
      knownTeams.forEach(t => { const el=document.createElement("option"); el.value=t; teamDatalist.appendChild(el); });
    }
  }
  function fillNameOptions() {
    if(preceptPartnerOptions) preceptPartnerOptions.innerHTML="";
    if(avoidOptions) avoidOptions.innerHTML="";
    const seen=new Set();
    nursesCache.forEach(n => {
      const nm = n.name||"";
      if(nm && !seen.has(nm)) {
        seen.add(nm);
        if(preceptPartnerOptions) preceptPartnerOptions.appendChild(new Option(nm));
        if(avoidOptions) avoidOptions.appendChild(new Option(nm));
      }
    });
  }

  function setupMultiNameAutocomplete(inputEl) {
    if (!inputEl || inputEl.dataset.multiNameInited === "1") return;
    inputEl.dataset.multiNameInited = "1";
    const parent = inputEl.parentElement || inputEl;
    if (getComputedStyle(parent).position === "static") parent.style.position = "relative";
    const panel = document.createElement("div"); panel.className = "name-suggestions hidden"; parent.appendChild(panel);

    function updatePanel() {
      const allNames = nursesCache.map(n => n.name).filter(Boolean);
      const value = inputEl.value;
      const lastComma = value.lastIndexOf(",");
      const token = (lastComma >= 0 ? value.slice(lastComma + 1) : value).trimStart();
      if (!token) { panel.innerHTML = ""; panel.classList.add("hidden"); return; }
      const lower = token.toLowerCase();
      const candidates = [...new Set(allNames)].filter((name) => name.toLowerCase().includes(lower)).slice(0, 10);
      if (!candidates.length) { panel.innerHTML = ""; panel.classList.add("hidden"); return; }
      panel.innerHTML = "";
      candidates.forEach((name) => {
        const item = document.createElement("div"); item.className = "name-suggestion-item"; item.textContent = name;
        item.addEventListener("mousedown", (e) => {
          e.preventDefault();
          const prefix = lastComma >= 0 ? value.slice(0, lastComma + 1) : "";
          inputEl.value = prefix ? `${prefix} ${name}` : name;
          panel.classList.add("hidden");
          setTimeout(autoCalculateLevel, 100);
        });
        panel.appendChild(item);
      });
      panel.classList.remove("hidden");
    }
    inputEl.addEventListener("input", updatePanel);
    inputEl.addEventListener("focus", updatePanel);
    inputEl.addEventListener("blur", () => setTimeout(()=>panel.classList.add("hidden"), 150));
  }
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("name-suggestion-item")) setTimeout(autoCalculateLevel, 100);
  });

  teamInput.addEventListener("blur", () => {
    const v = teamInput.value.trim();
    if (v && !knownTeams.has(v)) {
      if(confirm(`새 팀 '${v}'을(를) 생성하시겠습니까?`)) { knownTeams.add(v); fillTeamOptions(); }
      else { teamInput.value = ""; }
    }
  });

  // -----------------------------------------------------------
  // Table Rendering (Updated Columns)
  // -----------------------------------------------------------
  function renderNurseTable() {
    if (!nurseTableBody) return;
    nurseTableBody.innerHTML = "";

    const sorted = [...nursesCache].sort((a, b) => {
      const dir = sortState.dir === "asc" ? 1 : -1;
      let va, vb;
      if (sortState.col === "staff_id") {
         const aN = parseInt(a.staff_id,10), bN = parseInt(b.staff_id,10);
         va = isNaN(aN) ? a.staff_id : aN; vb = isNaN(bN) ? b.staff_id : bN;
      } else if (sortState.col === "years_exp") { va = a.years_exp; vb = b.years_exp;
      } else if (sortState.col === "team") { va = a.team; vb = b.team;
      } else { va = a.name; vb = b.name; }
      return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
    });

    sorted.forEach((n) => {
      const tr = document.createElement("tr");
      tr.style.cursor = "pointer";
      if (selectedNurseId && String(n.nurse_id) === String(selectedNurseId)) tr.classList.add("selected-row");
      tr.addEventListener("click", () => editNurse(n, false));

      const td = (txt) => { const el = document.createElement("td"); el.textContent = txt; return el; };

      tr.appendChild(td(n.staff_id));
      tr.appendChild(td(n.name));
      const tdY = td(n.years_exp); if(n.join_year) tdY.title = `입사: ${n.join_year}`;
      tr.appendChild(tdY);
      tr.appendChild(td(n.team));

      // Leader
      const l = [];
      if(n.leader_d) l.push("D");
      if(n.leader_e) l.push("E");
      if(n.leader_n) l.push("N");
      tr.appendChild(td(l.join("/") || "-"));

      tr.appendChild(td(n.precept_partner ? stripIdFromLabel(n.precept_partner) : "-"));
      tr.appendChild(td(stripIdFromLabel(n.avoid_list) || "-"));
      tr.appendChild(td(n.relative_work_days !== 1.0 ? n.relative_work_days : "-"));

      const tags = (n.tags || []).map(t => {
        if(t==='pregnant') return '임산부';
        if(t==='night_keep') return 'N전담';
        if(t==='fixed_day') return 'D전담';
        if(t==='fixed_evening') return 'E전담';
        return t;
      });
      const prefs = (n.preferred_shifts || []).map(s => `+${s}`);
      const avoids = (n.avoid_shifts || []).map(s => `-${s}`);

      const combined = [...tags, ...prefs, ...avoids].join(", ");
      tr.appendChild(td(combined || "-"));

      const tdEdit = document.createElement("td");
      const btn = document.createElement("button");
      btn.textContent = "수정"; btn.className = "secondary";
      btn.addEventListener("click", (e) => { e.stopPropagation(); editNurse(n, true); });
      tdEdit.appendChild(btn);
      tr.appendChild(tdEdit);

      nurseTableBody.appendChild(tr);
    });
  }

  // Header Sort Click
  if(nurseTableHead) {
    nurseTableHead.addEventListener("click", (e) => {
        const th = e.target.closest("th");
        if(!th) return;
        const col = th.dataset.col;
        if(!col || col==="edit" || col==="tags_prefs") return;
        if(sortState.col === col) sortState.dir = sortState.dir==="asc" ? "desc" : "asc";
        else { sortState.col = col; sortState.dir = "asc"; }
        renderNurseTable();
    });
  }

  // -----------------------------------------------------------
  // CRUD
  // -----------------------------------------------------------
  async function loadNurses() {
    try {
      const res = await fetch("/api/nurses");
      nursesCache = (await res.json()).map(normalizeNurse);
      fillTeamOptions(); fillNameOptions();
      setupMultiNameAutocomplete(preceptPartnerInput);
      setupMultiNameAutocomplete(avoidInput);
      renderNurseTable();
    } catch (e) { showFormMessage("로드 실패", true); }
  }

  function editNurse(n, scroll) {
    selectedNurseId = n.nurse_id;
    nurseIdInput.value = n.nurse_id || "";
    staffIdInput.value = n.staff_id;
    nameInput.value = n.name;
    joinYearInput.value = n.join_year || currentYear;
    yearsExpInput.value = n.years_exp;

    const lvl = n.level;
    const match = ["1.0","0.7","1.1","1.2"].find(v => Math.abs(parseFloat(v)-lvl)<0.01);
    levelSelect.value = match || "manual";
    updateLevelCustomVisibility();
    if(!match) levelCustomInput.value = lvl;

    leaderDInput.checked = !!n.leader_d;
    leaderEInput.checked = !!n.leader_e;
    leaderNInput.checked = !!n.leader_n;
    relWorkInput.value = n.relative_work_days;

    tagCheckboxes.forEach(cb => cb.checked = n.tags.includes(cb.value));
    prefCheckboxes.forEach(cb => cb.checked = n.preferred_shifts.includes(cb.value));
    avoidCheckboxes.forEach(cb => cb.checked = n.avoid_shifts.includes(cb.value));
    updateTagInteractions();

    teamInput.value = n.team;
    preceptPartnerInput.value = stripIdFromLabel(n.precept_partner);
    avoidInput.value = stripIdFromLabel(n.avoid_list);

    setStaffDupState(false, "");
    if(scroll) window.scrollTo({top:0, behavior:"smooth"});
    if(deleteNurseBtn) deleteNurseBtn.style.display = "inline-block";
    saveNurseBtn.textContent = "수정 저장";
  }

  function resetForm() {
    nurseForm.reset();
    joinYearInput.value = currentYear; yearsExpInput.value = "0";
    levelSelect.value = "1.0"; updateLevelCustomVisibility();
    selectedNurseId = null;
    updateTagInteractions();
    if(deleteNurseBtn) deleteNurseBtn.style.display = "none";
    saveNurseBtn.textContent = "저장";
  }

  resetFormBtn.addEventListener("click", resetForm);

  nurseForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const tags = Array.from(tagCheckboxes).filter(c=>c.checked).map(c=>c.value);
    const pref = Array.from(prefCheckboxes).filter(c=>c.checked).map(c=>c.value);
    const avd = Array.from(avoidCheckboxes).filter(c=>c.checked).map(c=>c.value);

    // [Fix] leader_eligible 객체와 개별 필드 모두 전송
    const leaderEligible = {
        D: leaderDInput.checked,
        E: leaderEInput.checked,
        N: leaderNInput.checked
    };

    const payload = {
        staff_id: staffIdInput.value.trim(),
        name: nameInput.value.trim(),
        years_exp: parseInt(yearsExpInput.value)||0,
        join_year: parseInt(joinYearInput.value)||null,
        level_weight: parseFloat(levelSelect.value==="manual"?levelCustomInput.value:levelSelect.value)||1.0,
        relative_work_days: parseFloat(relWorkInput.value)||1.0,

        leader_eligible: leaderEligible,
        leader_d: leaderDInput.checked,
        leader_e: leaderEInput.checked,
        leader_n: leaderNInput.checked,

        team_code: teamInput.value.trim(),
        tags: tags,
        preferred_shifts: pref,
        avoid_shifts: avd,
        precept_partner: preceptPartnerInput.value.trim(),
        avoid_list: avoidInput.value.trim()
    };

    try {
        const nid = nurseIdInput.value;
        const res = await fetch(nid ? `/api/nurses/${nid}` : "/api/nurses", {
            method: nid ? "PUT" : "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });
        if(!res.ok) throw new Error("Save failed");
        showFormMessage("저장 완료");
        loadNurses(); resetForm();
    } catch(e) { showFormMessage("저장 실패", true); }
  });

  downloadNursesBtn.addEventListener("click", () => { window.location.href = "/api/nurses/download"; });
  uploadExcelBtn.addEventListener("click", async () => {
    if (!excelFileInput.files[0]) return showUploadMessage("파일 선택 필요", true);
    const fd = new FormData(); fd.append("file", excelFileInput.files[0]);
    try {
        const res = await fetch("/api/upload_excel", { method: "POST", body: fd });
        if(!res.ok) throw new Error();
        const d = await res.json();
        showUploadMessage(`완료: ${d.created} 추가, ${d.updated} 수정`);
        loadNurses();
    } catch(e) { showUploadMessage("업로드 실패", true); }
  });

  async function loadWardInfo() {
    if (!wardNameDisplay) return;
    try {
      const res = await fetch("/api/ward");
      if (res.ok) {
        const data = await res.json();
        const wName = data.ward_name || data.name || "";
        if (wName) {
           wardNameDisplay.textContent = `[${wName}]`;
        }
      }
    } catch (e) {
      console.log("Ward info load failed or not supported.");
    }
  }

  loadNurses();
  loadWardInfo();
});