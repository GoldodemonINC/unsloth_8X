# NRC-3 License, version 1.0

**Non-Rebranding Commons**

Copyright 2026-present the X8 authors. All rights reserved except as granted below.

---

## Plain-language summary (not binding)

- Use it, change it, run it in private, and use it inside a company. No fee, no permission needed.
- Share the source, publish your fork, teach with it, write about it, mirror it for research.
- Keep this notice, keep the file headers, and say what you changed.
- Do not relaunch it as your own product: no new name, logo or interface, no near-miss
  name, no "official" or "endorsed" claim.
- Do not run it as a hosted service for other people, ship packaged builds to the public, or
  fold it into something you sell, without asking first.
- Model weights, tokenizers, datasets and inherited code keep their own licenses. This one
  does not reach them, and it does not put extra conditions on anyone who receives them.

The sections below are the license. Where the summary and a section disagree, the section wins.

---

## 1. What this license covers

This license covers the **Original Additions**:

1. each file in this project whose header reads `SPDX-License-Identifier: NRC-3`;
2. any file that a `NOTICE-NRC-3` file at the project root names as covered;
3. the run records this project writes — the `license` block of `run_manifest.json`, the
   `LICENSE-NRC-3.txt` notice written beside a produced checkpoint, and the `license` field of a
   saved pipeline.

Those files and records are the **Software**.

Everything else in this repository keeps its own license. Code inherited from Unsloth is
Apache-2.0 and the Studio application is AGPL-3.0; both are distributed under their own terms
and are not relicensed here. A file that carries an Apache-2.0 or AGPL-3.0 header is not covered
by this license, whatever its provenance and whoever edited it. Model weights, tokenizers and
datasets keep whatever license came with them from their source.

If you cannot tell whether a file is covered, it is not. Ask before relying on it.

## 2. Definitions

**You** means the individual or entity exercising rights under this license.

**Modified Version** means the Software changed in any way, in whole or in part, including a
version combined with other code.

**Distribute** means to give, send, publish, host or otherwise make the Software available to
anyone other than You, including through a download page, a registry, an app store, a container
image, a shipped device or a network service. **Distribution** means the same thing.

**Your Organisation** means You and the employees, contractors and systems acting for You inside
one legal entity, plus your own household.

**Public Distribution** means Distribution to anyone outside Your Organisation.

**Packaged Distribution** means Public Distribution in a form a recipient runs without building
it: an installer, a binary, a container or machine image, a packaged checkpoint or adapter
folder, or anything published to a package registry, app store, marketplace or download page.

**At Scale** means any one of: Public Distribution to more than one hundred recipients in any
twelve-month period; any Packaged Distribution; any Hosted Offering.

**Hosted Offering** means operating the Software, or a Modified Version, so that anyone outside
Your Organisation can use it over a network — a web interface, an API, a bot, a plugin, a
fine-tuning or inference service — whether or not money changes hands.

**Commercial Bundle** means Distribution as part of a product, service, hardware, support plan
or platform for which You charge, or which You supply to customers as part of what they paid for.

**Rebranded Offering** means presenting the Software, or a Modified Version, to others as your
own product or service, where **any** of these is true:

1. it carries a different product name, logo, icon or user interface;
2. it carries the project's name, or a name confusingly similar to it, with a qualifier, edition,
   suffix or prefix that suggests it is a version of the project;
3. it is described as official, certified, endorsed, partnered or authorised, or is presented in a
   way a reasonable person would read that way.

**Entity** means a company, partnership, non-profit, public body or any other organised group.

**Permission** means prior written permission from the copyright holders, given for a named use.
Permission is not implied by silence, by a previous grant to someone else, or by the absence of a
reply.

## 3. What you may do

Subject to sections 4 and 5, this license grants You a worldwide, royalty-free, non-exclusive
license to:

1. **Use** the Software for any purpose, including commercial use and internal business use,
   on machines You control.
2. **Modify** the Software and keep your changes private. Private changes carry no obligation to
   publish anything.
3. **Distribute** the Software and Modified Versions, subject to section 4.
4. **Publish the source** of a Modified Version on a public code host, so others can read, learn
   from and build on it, provided the distribution is not a Rebranded Offering, not a Hosted
   Offering, not a Commercial Bundle, and keeps the notices and headers intact.
5. **Combine** the Software with other code, and distribute the combination, provided the
   Software part keeps this license and its notices, and the combination is not itself a use
   reserved by section 5.
6. **Contribute** changes back to the project. Contributions are accepted under this license.

## 4. Conditions when you distribute

These apply when You distribute, not when You use the Software privately.

1. **Keep the notices.** Include this license, the copyright line and any `NOTICE-NRC-3` file
   with every copy You distribute.
2. **Keep the headers.** Do not remove, rewrite or obscure the `SPDX-License-Identifier` line or
   the copyright line in any covered file, and do not strip the `license` block from a run record
   or the notice from a produced folder.
3. **Say what changed.** Add a short note to a Modified Version describing the changes and their
   date. A changelog line or a header comment is enough.
4. **Do not broaden the grant.** You may not license the Software, or a Modified Version of it,
   under terms that give recipients more freedom than this license does, or that drop the
   reservations in section 5. You may license your own separate code, your models and your data
   however You like, and You may add stricter terms to your own work.
5. **Keep Modified Versions under NRC-3.** A Modified Version You distribute publicly stays under
   this license. Only the Software and its modifications carry NRC-3 forward.
6. **Do not imply endorsement.** You may state that your work is based on this project, but not
   that the project or its authors endorse You or your use.

## 5. What you may not do without Permission

These are reserved. Each one needs Permission.

1. **No Rebranded Offering.** You may not make a Rebranded Offering of the Software or a
   Modified Version.
2. **No Hosted Offering.** You may not operate the Software or a Modified Version as a Hosted
   Offering for anyone outside Your Organisation.
3. **No At-Scale Public Distribution.** You may not carry out At-Scale Public Distribution.
   This applies to You as an individual exactly as it applies to a company, so one developer
   shipping a public build needs Permission the same way an organisation does.
4. **No Commercial Bundle.** You may not include the Software or a Modified Version in a
   Commercial Bundle.
5. **No relicensing and no notice stripping.** You may not relicense the Software or a Modified
   Version under other terms, and You may not remove or obscure this license, a file header, a
   run-record `license` block, or a `LICENSE-NRC-3.txt` notice.
6. **No false claims.** You may not describe a Modified Version as official, endorsed or
   certified, and You may not attribute benchmark, evaluation or training results to this project
   that it did not produce.

For clarity, none of these need Permission: using the Software privately or inside Your
Organisation, including to run your own business; modifying it and keeping the changes private;
publishing your changed source on a code host under this license with the notices kept; sharing
a patch; teaching with it; writing about it; mirroring it for research; and running it on
machines You control for your own work.

Asking for Permission is a message to the maintainers. Silence is not Permission.

No trademark or logo rights are granted. The project's names and marks stay with their owners,
and section 5.1 applies to them whether or not a trademark is registered.

## 6. Relationship to other licenses

This license does not change, weaken or replace any other license that applies to a file, a
dependency or a model in this repository. If this license conflicts with a license that governs
inherited code, that other license governs that code, and this one continues to govern the
Original Additions.

Distributing the Software as part of a work that also contains AGPL-3.0 code does not put the
Original Additions under the AGPL. The reverse holds too: the reservations in section 5 apply to
the Original Additions and the run records, and they do not add conditions to code that is
governed by the AGPL or by Apache-2.0. A recipient's rights in those parts come from those
licenses alone, and You remain responsible for satisfying each license that applies to what You
distribute.

This repository as distributed here carries NRC-3 for the Original Additions, Apache-2.0 for the
Unsloth code it inherits and AGPL-3.0 for the Studio application. Vendored libraries, generated
files and dependencies a user installs separately are not covered by this license and do not
change what it covers.

## 7. Contributions and patents

By contributing to the Software You agree that your contribution is licensed under NRC-3, with no
additional terms, and that You have the right to grant that license.

You also grant the copyright holders a perpetual, irrevocable, worldwide, royalty-free,
sublicensable license to your contribution, so the project can relicense, reword and re-release
the Software as a whole without tracking down every contributor.

Each contributor grants You a royalty-free patent license covering patent claims they own or
control that are necessarily infringed by their contribution alone. That patent license ends if
You bring a patent action alleging that the Software infringes a patent.

## 8. Term and termination

This license runs for as long as the copyright in the Software lasts, unless it ends earlier
under this section.

Rights under section 5 end **immediately**, without notice or cure period, if You carry out a use
described in sections 5.1 to 5.4 without Permission. On that ending You must stop the
distribution, withdraw the offering, and take down the copies You published.

Rights under section 4 end if You breach a condition there and do not cure the breach within
thirty days of being told about it in writing. During those thirty days You may not continue the
distribution that breached the condition.

A cured breach reinstates your rights. A second breach of the same kind does not.

Ending this license does not affect the rights of anyone who received the Software from You in
compliance with it, and does not affect any license that applies to code this license does not
cover. Sections 1, 5, 6, 7, 9, 10 and 11 survive termination.

## 9. No warranty, limited liability

The Software is provided **as is**, without warranty of any kind, express or implied, including
merchantability, fitness for a particular purpose, title and non-infringement. You use it at your
own risk. Nothing here obliges the copyright holders to provide support, updates, fixes or a
response to a Permission request.

To the fullest extent the law allows, the copyright holders are not liable for any claim, damage
or other liability arising from the Software or your use of it, whether in contract, tort or
otherwise.

## 10. Versions

This is version 1.0 of NRC-3. The name marks the generation: NRC-1 reserved two things, NRC-2
(version 2.0) reserved three, and this one reserves five and closes the gaps the earlier two left
open. The version count restarts at 1.0 so that the name and the terms stay in step; an SPDX tag
says which generation You are dealing with without a lookup.

Compared with NRC-2 this revision changes: the reservations bind any distributor rather than
companies alone, a hosted offering needs Permission whatever it is called, a commercial bundle
needs Permission, notices and headers may not be stripped, the licence-count cap is gone in
favour of packaged builds and commercial bundles, a breach of the core reservations ends the
license immediately instead of after a grace period, and contributors grant the project the
right to relicense.

You may use the version You received with the Software. A later version applies to the Software
only if You choose to take it, and never retroactively to a distribution You completed under an
earlier version.

## 11. General

If a provision is held unenforceable, it is limited to the minimum extent needed and the rest of
the license stands.

A failure to enforce a provision is not a waiver of it. A waiver is only effective in writing and
only for the instance it names.

You may not transfer this license separately from the material it covers. A transfer of the whole
of Your business carries it along.

This license is the entire agreement about the Software between You and the copyright holders.

It is governed by the laws of the copyright holders' principal place of business, excluding that
place's conflict-of-law rules, and the courts there have exclusive jurisdiction. Nothing here
removes a right You have under mandatory law where You live.

## Applying this license

To place a file under NRC-3, add these two lines at the top of it:

```
SPDX-License-Identifier: NRC-3
Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md
```

Files that keep an Apache-2.0 or AGPL-3.0 header are not covered by this license, whatever their
provenance.
