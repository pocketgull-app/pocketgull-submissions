# Kaggle Notebook & Submission Markdown Presentation Template
*Derived from Phil Gear's [3D Knee Anatomy Simulator and Waiting Game](https://www.kaggle.com/code/philgear/3d-knee-anatomy-simulator-and-waiting-game)*

This template establishes the official presentation standard for all Kaggle notebooks, competition milestones, and submission descriptions across PocketGull projects.

---

## 1. Master Obsidian/Navy UI Palette

| Token | Hex Code | Visual Meaning & Role |
| :--- | :--- | :--- |
| **Surface Base** | `#0d1522` | Card container background (Dark obsidian-navy) |
| **Surface Inset** | `#090d16` | Nested inner bays, terminal screens, badge backings |
| **Border Subtle** | `#1e293b` | 1px border frames and horizontal divider lines |
| **Cyan Rail** | `#06b6d4` | Primary brand accent rail, Module 01, lead badges |
| **Sky Blue** | `#38bdf8` | Invariant 01, interactive links, secondary badges |
| **Rose / Coral**| `#fb7185` | Invariant 02, biological stress, chemical cleavage |
| **Amber Gold** | `#f59e0b` | Kinetics, module 02, warning thresholds |
| **Emerald** | `#10b981` | GreenOps, verified benchmarks, healthy assertions |
| **Purple / Violet**| `#a855f7`| Deep learning, PyTorch encoders, mass transport |

---

## 2. Cell 0: Master Header Markdown Card

```html
<div style="background: #0d1522; color: #f8fafc; border-radius: 4px; padding: 26px 30px; border: 1px solid #1e293b; border-left: 4px solid #06b6d4; box-shadow: 0 12px 32px rgba(5,8,14,0.7); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.65; margin-bottom: 24px; font-feature-settings: 'tnum' 1, 'zero' 1;">

<!-- HEADER ROW: BADGES & WORKBENCH STAMP -->
<div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #1e293b;">
  <div>
    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #06b6d4; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 3px 9px; border-radius: 2px; border: 1px solid #1e293b;">{{TRACK_BADGE}}</span>
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #10b981; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 3px 9px; border-radius: 2px; border: 1px solid #1e293b;">{{BENCHMARK_OR_SCORE}}</span>
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #a855f7; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 3px 9px; border-radius: 2px; border: 1px solid #1e293b;">{{ARCHITECTURE_TAG}}</span>
    </div>
    <div style="font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.025em; line-height: 1.25;">
      {{NOTEBOOK_TITLE}}
    </div>
    <div style="font-size: 14.5px; color: #94a3b8; margin-top: 6px; font-weight: 500;">
      {{NOTEBOOK_SUBTITLE}}
    </div>
  </div>

  <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px; font-family: monospace; font-size: 11.5px; color: #64748b;">
    <div style="display: flex; align-items: center; gap: 6px; background: #090d16; padding: 4px 10px; border-radius: 2px; border: 1px solid #1e293b;">
      <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
      <span style="color: #cbd5e1; font-weight: 600;">{{PRIMARY_TOOL_NAME}}</span>
    </div>
    <div style="color: #64748b; font-size: 10.5px;">{{TARGET_RUNTIME}}</div>
  </div>
</div>

<!-- LINEAGE & AUTHOR ROW -->
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; padding: 10px 14px; background: #090d16; border: 1px solid #1e293b; border-radius: 3px; font-size: 12px; color: #94a3b8;">
  <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
    <span style="color: #f8fafc; font-weight: 600;">Author: Phil Gear (<a href="https://www.kaggle.com/philgear" style="color: #06b6d4; text-decoration: none;">@philgear</a>)</span>
    <span>•</span>
    <span><a href="https://font.pocketgull.app" target="_blank" style="color: #06b6d4; text-decoration: none;">PocketGull Clinical Superfamily</a></span>
    <span>•</span>
    <span>{{FRAMEWORK_BADGE}}</span>
  </div>
</div>

<!-- EXECUTIVE DIAGNOSTIC SUMMARY -->
<div style="background: #090d16; border: 1px solid #1e293b; border-left: 3px solid #06b6d4; border-radius: 3px; padding: 16px 20px; margin-bottom: 22px;">
  <div style="font-size: 11px; font-family: monospace; font-weight: 700; color: #06b6d4; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px;">Executive Diagnostic Summary</div>
  <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6;">
    {{EXECUTIVE_SUMMARY_TEXT}}
  </div>
</div>

<!-- INVARIANTS GRID (2 COLUMNS) -->
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 24px;">
  <div style="background: #090d16; border: 1px solid #1e293b; border-top: 2px solid #38bdf8; border-radius: 3px; padding: 16px;">
    <div style="font-family: monospace; font-size: 11px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">INVARIANT 01 • {{INVARIANT_1_LABEL}}</div>
    <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">{{INVARIANT_1_TITLE}}</div>
    <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.55;">
      {{INVARIANT_1_BODY}}
    </div>
  </div>

  <div style="background: #090d16; border: 1px solid #1e293b; border-top: 2px solid #fb7185; border-radius: 3px; padding: 16px;">
    <div style="font-family: monospace; font-size: 11px; color: #fb7185; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">INVARIANT 02 • {{INVARIANT_2_LABEL}}</div>
    <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">{{INVARIANT_2_TITLE}}</div>
    <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.55;">
      {{INVARIANT_2_BODY}}
    </div>
  </div>
</div>

<!-- SYSTEM MODULES GRID -->
<div style="margin-bottom: 20px;">
  <div style="font-size: 12px; font-family: monospace; font-weight: 700; letter-spacing: 0.1em; color: #06b6d4; text-transform: uppercase; margin-bottom: 12px;">
    DOUBLE FEATURE ARCHITECTURE • {{N_MODULES_COUNT}} SYSTEM MODULES
  </div>
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #06b6d4;">
      <div style="font-family: monospace; font-size: 10.5px; color: #06b6d4; font-weight: 700;">MODULE 01</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">{{MOD1_TITLE}}</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">{{MOD1_DESC}}</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #f59e0b;">
      <div style="font-family: monospace; font-size: 10.5px; color: #f59e0b; font-weight: 700;">MODULE 02</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">{{MOD2_TITLE}}</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">{{MOD2_DESC}}</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #38bdf8;">
      <div style="font-family: monospace; font-size: 10.5px; color: #38bdf8; font-weight: 700;">MODULE 03</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">{{MOD3_TITLE}}</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">{{MOD3_DESC}}</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #10b981;">
      <div style="font-family: monospace; font-size: 10.5px; color: #10b981; font-weight: 700;">MODULE 04</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">{{MOD4_TITLE}}</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">{{MOD4_DESC}}</div>
    </div>
  </div>
</div>

<!-- GREENOPS & TOKEN METRICS FOOTER -->
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding-top: 14px; border-top: 1px solid #1e293b; font-family: monospace; font-size: 11.5px; color: #64748b;">
  <div style="display: flex; align-items: center; gap: 6px;">
    <span style="color: #10b981;">GREENOPS:</span>
    <span>{{GREENOPS_TEXT}}</span>
  </div>
  <div style="display: flex; align-items: center; gap: 6px;">
    <span style="color: #06b6d4;">POCKETGULL TOKENS:</span>
    <span>--card-bg: #0d1522 • --surface-inner: #090d16</span>
  </div>
</div>

</div>
```

---

## 3. Section Divider & Code Module Card

```html
<div style="background: #0d1522; color: #f8fafc; border-radius: 4px; padding: 26px 30px; border: 1px solid #1e293b; border-left: 4px solid {{ACCENT_COLOR_HEX}}; box-shadow: 0 12px 32px rgba(5,8,14,0.7); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.65; margin: 18px 0;">

<div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; padding-bottom: 14px; border-bottom: 1px solid #1e293b;">
  <div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: {{ACCENT_COLOR_HEX}}; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 2px 8px; border-radius: 2px; border: 1px solid #1e293b;">SECTION {{SECTION_NUMBER}}</span>
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #10b981; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 2px 8px; border-radius: 2px; border: 1px solid #1e293b;">{{STATUS_BADGE}}</span>
    </div>
    <div style="font-size: 21px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-top: 6px;">{{SECTION_TITLE}}</div>
    <div style="font-size: 12.5px; color: #94a3b8; margin-top: 4px;">{{SECTION_SUBTITLE}}</div>
  </div>
  <div style="display: flex; align-items: center; gap: 6px; font-family: monospace; font-size: 11px; color: #64748b; background: #090d16; padding: 4px 10px; border-radius: 2px; border: 1px solid #1e293b;">
    <span style="display: inline-block; width: 6px; height: 6px; background: {{ACCENT_COLOR_HEX}};"></span>
    <span>{{STAGE_TAG}}</span>
  </div>
</div>

<div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin-bottom: 16px;">
  {{SECTION_NARRATIVE_DESCRIPTION}}
</div>

</div>
```

---

## 4. Closing Card: Discussion & Community Collaboration Footer

```html
<div style="background: #0d1522; color: #f8fafc; border-radius: 4px; padding: 26px 30px; border: 1px solid #1e293b; border-left: 4px solid #06b6d4; box-shadow: 0 12px 32px rgba(5,8,14,0.7); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.65; margin: 18px 0; font-feature-settings: 'tnum' 1, 'zero' 1;">

<div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; padding-bottom: 14px; border-bottom: 1px solid #1e293b;">
  <div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #06b6d4; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 2px 8px; border-radius: 2px; border: 1px solid #1e293b;">COLLABORATION</span>
    </div>
    <div style="font-size: 21px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-top: 6px;">Join the Discussion & Connect</div>
    <div style="font-size: 12.5px; color: #94a3b8; margin-top: 4px; font-weight: 500;">{{COMMUNITY_SUBTITLE}}</div>
  </div>
  <div style="display: flex; align-items: center; gap: 6px; font-family: monospace; font-size: 11px; color: #64748b; background: #090d16; padding: 4px 10px; border-radius: 2px; border: 1px solid #1e293b;">
    <span style="display: inline-block; width: 6px; height: 6px; background: #06b6d4;"></span>
    <span>SPECIMEN BENCH</span>
  </div>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-bottom: 18px;">
  <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 2px solid #06b6d4;">
    <div style="font-family: monospace; font-size: 10.5px; color: #06b6d4; font-weight: 700;">TOPIC 01</div>
    <div style="font-weight: 700; color: #ffffff; font-size: 12.5px; margin: 2px 0;">{{TOPIC_1_TITLE}}</div>
    <div style="font-size: 11.5px; color: #94a3b8;">{{TOPIC_1_QUESTION}}</div>
  </div>
  <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 2px solid #38bdf8;">
    <div style="font-family: monospace; font-size: 10.5px; color: #38bdf8; font-weight: 700;">TOPIC 02</div>
    <div style="font-weight: 700; color: #ffffff; font-size: 12.5px; margin: 2px 0;">{{TOPIC_2_TITLE}}</div>
    <div style="font-size: 11.5px; color: #94a3b8;">{{TOPIC_2_QUESTION}}</div>
  </div>
  <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 2px solid #f59e0b;">
    <div style="font-family: monospace; font-size: 10.5px; color: #f59e0b; font-weight: 700;">TOPIC 03</div>
    <div style="font-weight: 700; color: #ffffff; font-size: 12.5px; margin: 2px 0;">{{TOPIC_3_TITLE}}</div>
    <div style="font-size: 11.5px; color: #94a3b8;">{{TOPIC_3_QUESTION}}</div>
  </div>
</div>

<div style="text-align: center; padding: 16px; background: #090d16; border: 1px solid #1e293b; border-radius: 3px; border-top: 2px solid #06b6d4;">
  <div style="font-size: 14px; font-weight: 800; color: #38bdf8; margin-bottom: 4px; font-family: monospace;">
    [BENCH RECORD COMPLETED] If you found this specimen analysis valuable, please upvote!
  </div>
  <div style="font-size: 11.5px; color: #64748b; font-family: monospace;">
    Author: Phil Gear (<a href="https://www.kaggle.com/philgear" style="color: #06b6d4; text-decoration: none;">@philgear</a>) • {{COMPETITION_TITLE}}
  </div>
</div>

</div>
```

---

## 5. Kaggle Submission Description Format

When submitting to Kaggle, write the submission note in this exact format:

```text
v{{VERSION}} (Kernel #{{KERNEL_NUMBER}}): SOTA {{LB_SCORE}} | {{KEY_CHANGE_1}} + {{KEY_CHANGE_2}} + {{KEY_CHANGE_3}} | PocketGull Neuro-Ergonomics
```
