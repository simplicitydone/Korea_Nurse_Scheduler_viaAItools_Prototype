// static/js/score_rules_panel.js

const ScoreRulesPanel = (() => {
  // [Fix] 중복 선언 제거 및 통합
  const state = {
    wardId: null,
    rules: [],
    fairness_weights: { den_balance: 30.0, off_balance: 20.0, weekend_holiday_balance: 40.0 },
    saveTimer: null
  };

  const STRENGTH_MAP = {
      "weak": { label: "약", weight: 5.0, color: "#a5d6a7" },
      "medium": { label: "중", weight: 10.0, color: "#ffe082" },
      "strong": { label: "강", weight: 30.0, color: "#ffab91" },
      "critical": { label: "최대", weight: 80.0, color: "#ef9a9a" }
  };

  const RULE_NAMES = {
      "CONSEC_WORK_5_PEN": "5일 연속 근무 제한",
      "WORK_O_WORK_PEN": "퐁당퐁당 방지",
      "N_COUNT_MAX_16": "월 나이트 횟수 제한",
      "pattern_nod": "N-O-D 패턴 방지",
      "consecutive_night": "연속 나이트 제한",
      "avoid_ne": "N -> E 패턴 방지"
  };

  function _container() { return document.getElementById("score-rules-panel"); }

  function _weightToStrength(w) {
      if (w <= 7.0) return "weak";
      if (w <= 20.0) return "medium";
      return "strong";
  }

  async function init(wardId) {
      state.wardId = wardId;
      await fetchConfig();
      render();
  }

  async function fetchConfig() {
      try {
          const res = await fetch(`/api/score-config/active?ward_id=${state.wardId}`);
          if(res.ok) {
              const data = await res.json();
              if(data.config_json) {
                  state.rules = data.config_json.rules || [];
                  if(data.config_json.fairness_weights) {
                      state.fairness_weights = { ...state.fairness_weights, ...data.config_json.fairness_weights };
                  }
              }
          }
      } catch(e) { console.error("Config Load Fail", e); }
      render();
  }

  function render() {
    const el = _container();
    if (!el) return;

    let html = `
      <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <h4 style="margin:0;">⚙️ 세부 규칙 설정 (자동 저장됨)</h4>
        <button class="secondary small" onclick="ScoreRulesPanel.openAddRuleModal()">+ 규칙 추가</button>
      </div>

      <div id="active-rules-list" class="rules-list"></div>

      <!-- 공정성 설정 -->
      <div class="fairness-section" style="margin-top:15px; padding-top:10px; border-top:1px dashed #ddd;">
        <h5 style="margin:0 0 5px 0;">📊 공정성 균형 설정</h5>
        ${_renderFairnessControls()}
      </div>

      <!-- Add Rule Modal -->
      <div id="add-rule-ui" class="add-rule-box hidden" style="background:#f9f9f9; padding:10px; border:1px solid #ddd; margin-top:10px; border-radius:5px;">
        <div style="display:flex; gap:5px; margin-bottom:5px;">
            <select id="new-rule-type" onchange="ScoreRulesPanel.onTypeChange()" style="padding:4px;">
                <option value="pattern">패턴 가감점 (예: O->N)</option>
                <option value="consecutive">연속성 제한</option>
                <option value="count">총 개수 제어</option>
            </select>
            <input type="text" id="new-rule-name" placeholder="규칙 이름" style="flex:1; padding:4px;">
        </div>
        <div id="new-rule-params" style="margin-bottom:5px;"></div>
        <div style="display:flex; justify-content:flex-end; gap:5px;">
            <button class="secondary small" onclick="ScoreRulesPanel.closeAddRuleModal()">취소</button>
            <button class="primary small" onclick="ScoreRulesPanel.addRule()">추가</button>
        </div>
      </div>
    `;

    el.innerHTML = html;

    const listEl = el.querySelector("#active-rules-list");
    if(state.rules.length === 0) {
        listEl.innerHTML = "<div class='muted text-center small'>설정된 규칙이 없습니다.</div>";
    } else {
        state.rules.forEach((rule, idx) => {
            listEl.appendChild(_createRuleItem(rule, idx));
        });
    }
  }

  function _createRuleItem(rule, idx) {
      const div = document.createElement("div");
      div.className = "rule-item";
      div.style.cssText = "display:flex; align-items:center; justify-content:space-between; padding:8px; border-bottom:1px solid #eee;";

      const strengthKey = _weightToStrengthKey(rule.weight);

      div.innerHTML = `
        <div style="flex:1;">
            <div style="font-weight:600; font-size:0.9em;">${rule.id}</div>
            <div style="font-size:0.8em; color:#666;">${_formatRuleDesc(rule)}</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <label class="toggle-switch">
                <input type="checkbox" ${rule.enabled ? "checked" : ""} onchange="ScoreRulesPanel.toggleRule(${idx}, this.checked)">
                <span class="slider"></span>
            </label>
            ${_renderStrengthButtons(idx, strengthKey, 'rule')}
            <button class="text-btn danger" onclick="ScoreRulesPanel.deleteRule(${idx})">×</button>
        </div>
      `;
      return div;
  }

  function _renderFairnessControls() {
      const keys = [
          {k: "den_balance", label: "근무량 균형"},
          {k: "off_balance", label: "OFF 개수 균형"},
          {k: "weekend_holiday_balance", label: "주말/연휴 균형"}
      ];

      return keys.map(item => {
          const w = state.fairness_weights[item.k] || 10.0;
          const sKey = _weightToStrengthKey(w);
          return `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:0.85em;">${item.label}</span>
                ${_renderStrengthButtons(item.k, sKey, 'fairness')}
            </div>
          `;
      }).join("");
  }

  function _renderStrengthButtons(id, currentKey, type) {
      let html = `<div class="strength-group" style="display:flex; gap:2px;">`;
      ['weak', 'medium', 'strong'].forEach(k => {
          const isActive = k === currentKey;
          const color = STRENGTH_MAP[k].color;
          const style = isActive ? `background:${color}; border-color:${color}; font-weight:bold;` : `background:#f0f0f0;`;

          const clickFn = type === 'rule'
              ? `ScoreRulesPanel.updateRuleStrength(${id}, '${k}')`
              : `ScoreRulesPanel.updateFairnessStrength('${id}', '${k}')`;

          html += `<button type="button" style="padding:2px 6px; font-size:10px; border:1px solid #ccc; border-radius:3px; cursor:pointer; ${style}"
                   onclick="${clickFn}">${STRENGTH_MAP[k].label}</button>`;
      });
      html += `</div>`;
      return html;
  }

  function _formatRuleDesc(r) {
      if(r.type === 'sequence') {
          if(r.meta.kind === 'consecutive_work') return `연속 근무 ${r.meta.threshold}일 제한`;
          if(r.meta.kind === 'consecutive_night') return `연속 나이트 ${r.meta.threshold}일 제한`;
          if(r.pattern) return `패턴 '${r.pattern}' 방지`;
      }
      if(r.type === 'count') {
          return `${r.meta.shift} 근무 최대 ${r.max_value}회`;
      }
      return r.type;
  }

  function _weightToStrengthKey(w) {
      if(w <= 7) return 'weak';
      if(w <= 20) return 'medium';
      return 'strong';
  }

  // --- Interaction Logic ---

  function openAddRuleModal() {
      document.getElementById("add-rule-ui").classList.remove("hidden");
      onTypeChange();
  }
  function closeAddRuleModal() {
      document.getElementById("add-rule-ui").classList.add("hidden");
  }

  function onTypeChange() {
      const type = document.getElementById("new-rule-type").value;
      const container = document.getElementById("new-rule-params");

      const shifts = `<option value="D">Day</option><option value="E">Evening</option><option value="N">Night</option>
                      <option value="O">Off</option><option value="Work">근무(DEN)</option>`;

      let html = "";
      if(type === 'pattern') {
          html = `<input type="text" id="p-pattern" placeholder="패턴 (예: OD, NE)" style="width:100%; padding:4px;">`;
      } else if(type === 'consecutive') {
          html = `<select id="p-shift">${shifts}</select> 연속 <input type="number" id="p-threshold" value="3" style="width:40px;">일`;
      } else if(type === 'count') {
          html = `<select id="p-shift-c">${shifts}</select> 최대 <input type="number" id="p-max" value="5" style="width:40px;">회`;
      }
      container.innerHTML = html;
  }

  function addRule() {
      const name = document.getElementById("new-rule-name").value.trim();
      if(!name) return alert("이름을 입력하세요.");
      const type = document.getElementById("new-rule-type").value;

      let newRule = {
          id: name, enabled: true, weight: 10.0, scope: "nurse", meta: {}
      };

      if(type === 'pattern') {
          newRule.type = "sequence";
          newRule.pattern = document.getElementById("p-pattern").value.toUpperCase();
      } else if(type === 'consecutive') {
          newRule.type = "sequence";
          const s = document.getElementById("p-shift").value;
          if(s === "Work") { newRule.pattern = "CONSEC_WORK"; newRule.meta.kind = "consecutive_work"; }
          else if(s === "N") { newRule.pattern = "CONSEC_N"; newRule.meta.kind = "consecutive_night"; }
          else if(s === "O") { newRule.pattern = "CONSEC_O"; newRule.meta.kind = "consecutive_off"; }
          else { newRule.pattern = "CONSEC_" + s; }
          newRule.meta.threshold = parseInt(document.getElementById("p-threshold").value);
      } else if(type === 'count') {
          newRule.type = "count";
          newRule.max_value = parseInt(document.getElementById("p-max").value);
          newRule.meta.shift = document.getElementById("p-shift-c").value;
      }

      state.rules.push(newRule);
      closeAddRuleModal();
      render();
      autoSave();
  }

  function deleteRule(idx) {
      if(confirm("이 규칙을 삭제하시겠습니까?")) {
          state.rules.splice(idx, 1);
          render();
          autoSave();
      }
  }

  function toggleRule(idx, val) {
      state.rules[idx].enabled = val;
      autoSave();
  }

  function updateRuleStrength(idx, strKey) {
      state.rules[idx].weight = STRENGTH_MAP[strKey].weight;
      render();
      autoSave();
  }

  function updateFairnessStrength(key, strKey) {
      state.fairness_weights[key] = STRENGTH_MAP[strKey].weight;
      render();
      autoSave();
  }

  function autoSave() {
      if(state.saveTimer) clearTimeout(state.saveTimer);
      state.saveTimer = setTimeout(_saveToServer, 1000);
  }

  async function _saveToServer() {
      if(!state.wardId) return;
      const payload = {
          ward_id: state.wardId,
          version_label: `AutoSave_${Date.now()}`,
          config_json: getConfigJson(),
          active: true
      };
      try {
          await fetch("/api/score-config", {
              method: "POST", headers: {"Content-Type":"application/json"},
              body: JSON.stringify(payload)
          });
      } catch(e) { console.error("Auto-save failed", e); }
  }

  function getConfigJson() {
      return {
          version: "nsp_v4_auto",
          rules: state.rules,
          fairness_weights: state.fairness_weights
      };
  }

  return {
      init, getConfigJson,
      openAddRuleModal, closeAddRuleModal, onTypeChange, addRule,
      deleteRule, toggleRule, updateRuleStrength, updateFairnessStrength
  };
})();

window.ScoreRulesPanel = ScoreRulesPanel;