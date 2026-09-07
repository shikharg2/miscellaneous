Modern **embedded systems and Cyber-Physical Systems (CPS)** increasingly integrate real-time software, operating systems, sensors, actuators, communication networks, control logic, and physical processes. These systems are widely deployed across aerospace, automotive, industrial automation, robotics, energy, transportation, and critical infrastructure, where failures or compromises can affect not only computational systems but also the physical environment. Unlike conventional IT environments, embedded and CPS platforms are constrained by **real-time performance, deterministic execution, limited resources, reliability, safety, and physical-process integrity**.

A major challenge in securing such environments is the lack of an integrated framework that can first understand their normal runtime behavior, then correlate that behavior with known vulnerabilities and threat intelligence, and finally use this knowledge to autonomously perform controlled security assessments. Existing tools generally operate independently: performance-monitoring platforms observe CPU, memory, processes, scheduling, and network behavior; vulnerability databases provide static information on known weaknesses; and penetration-testing tools execute predefined actions with limited awareness of system context. This fragmentation makes it difficult to correlate low-level runtime behavior with vulnerabilities, attack paths, exploitability, and potential cyber-physical consequences.

The proposed work addresses this problem through a progressive three-stage framework.

### Stage 1 – Performance Testing and Observability Tool

The first stage will develop a unified **Performance Testing and Observability Tool** specifically targeting embedded and cyber-physical environments. The platform will collect and correlate runtime information across the **kernel, operating system, applications, processes, network stack, device drivers, hardware interfaces, and physical-system components**.

Technologies such as **eBPF** will be used on supported embedded Linux platforms to capture system calls, scheduler activity, interrupts, memory behavior, process interactions, networking events, and other low-level kernel telemetry. Network-analysis tools such as **Wireshark/TShark** and protocol-specific analyzers will provide communication-level visibility. Additional sources may include RTOS traces, application logs, hardware performance counters, CPU and memory utilization, execution latency, I/O behavior, driver events, sensor telemetry, actuator state, and communication protocols such as **CAN/CAN-FD, Ethernet, Modbus, OPC UA, MQTT, and other embedded or industrial protocols**.

The collected information will be normalized into a common telemetry model and temporally correlated to establish a complete representation of system behavior. This stage will create the baseline necessary to understand normal execution, detect deviations, identify bottlenecks, and determine how changes at one layer propagate through the rest of the cyber-physical system.

**Stage 1 Outcomes and Deliverables:**

* A unified performance-testing and observability platform for embedded and CPS environments.
* Kernel-, application-, network-, system-, hardware-, and device-level telemetry collection.
* Integration of eBPF, Wireshark/TShark, OS utilities, protocol analyzers, and platform-specific monitoring mechanisms.
* A normalized telemetry model for heterogeneous embedded-system data.
* Cross-layer event correlation and synchronized runtime timelines.
* Baseline performance profiles for selected embedded/CPS platforms.
* Detection of abnormal execution, communication, timing, and resource-consumption patterns.
* APIs and structured datasets that can be consumed by later security-analysis stages.

### Stage 2 – Automated Penetration Testing with CKG-Assisted Intelligence

The second stage will extend the observability platform into an **automated penetration-testing framework assisted by a Cybersecurity Knowledge Graph (CKG)**. The CKG will combine system information collected in Stage 1 with Cyber Threat Intelligence and vulnerability information including **CVE, CPE, CWE, CVSS, CAPEC, MITRE ATT&CK, MITRE ATT&CK for ICS, software and firmware versions, network topology, device relationships, attack prerequisites, and known exploitation techniques**.

The purpose of the knowledge graph will be to move beyond isolated vulnerability detection and provide contextual reasoning about the actual embedded or CPS environment. For example, a vulnerability will be associated not only with a CVE identifier but also with the affected device, software version, exposed interface, required privileges, communication path, potential attack technique, and possible consequences on the physical process.

Based on this information, penetration-testing frameworks such as **MITRE CALDERA, Metasploit**, vulnerability scanners, network-assessment utilities, protocol-testing tools, and other offensive-security frameworks will be orchestrated automatically. The CKG will assist in determining which tests are relevant, their prerequisites, potential attack sequences, and the expected impact of each action.

Runtime telemetry from Stage 1 will provide continuous feedback during execution. The system will therefore be able to observe whether a particular security action changes CPU utilization, process behavior, network activity, kernel state, response timing, sensor information, actuator behavior, or other CPS parameters.

This stage will establish a feedback loop consisting of:

**Discover → Correlate → Select Attack Path → Execute → Observe → Validate → Update Knowledge**

For safety-critical embedded and CPS environments, automated actions will additionally be constrained by predefined safety policies. High-risk actions can be limited to simulations, test benches, digital twins, hardware-in-the-loop environments, or explicitly authorized systems.

**Stage 2 Outcomes and Deliverables:**

* A CTI-driven Cybersecurity Knowledge Graph representing embedded/CPS assets, vulnerabilities, weaknesses, attack techniques, and system relationships.
* Automated mapping of observed hardware and software components to CPEs, CVEs, CWEs, CVSS scores, CAPEC patterns, and ATT&CK techniques.
* Automated attack-path generation using system topology, vulnerabilities, privileges, and attack prerequisites.
* Integration and orchestration of MITRE CALDERA, Metasploit, scanners, reconnaissance utilities, and domain-specific penetration-testing tools.
* CKG-assisted selection and prioritization of penetration-testing actions.
* Runtime validation of attack effects using Stage 1 telemetry.
* Structured recording of successful, unsuccessful, blocked, and partially successful attack sequences.
* Generation of attack graphs and evidence-backed penetration-testing reports.
* A continuously updated dataset containing system states, selected actions, tool commands, observations, attack results, and resulting system-state transitions.

### Stage 3 – Completely Autonomous Intelligent Penetration Testing using an SLM

The final stage will use the telemetry, attack paths, knowledge-graph relationships, tool outputs, and penetration-testing trajectories generated during the previous stages to develop a **Small Language Model-based autonomous penetration-testing system**.

The SLM will act as the intelligent reasoning and decision-making layer of the architecture. Rather than simply generating commands, it will operate in conjunction with the **observability platform, CKG, penetration-testing tools, and historical outcomes from Stage 2**.

The model will be trained or adapted using data such as system configurations, vulnerability relationships, CKG subgraphs, telemetry sequences, tool outputs, penetration-testing actions, successful and failed exploitation attempts, attack-path transitions, observed system responses, and remediation or defensive outcomes.

At runtime, the SLM will receive the current state of the embedded or CPS environment and autonomously determine how the assessment should proceed. It will be capable of querying the CKG, interpreting system telemetry, selecting security tools, constructing multi-stage penetration-testing strategies, evaluating intermediate results, abandoning ineffective attack paths, and choosing alternative actions.

The architecture will therefore transition from **rule-based and knowledge-assisted automation in Stage 2 to adaptive and autonomous reasoning in Stage 3**.

Instead of following predefined penetration-testing workflows, the system will progressively learn how combinations of vulnerabilities, configurations, system states, and previous actions affect the probability of successful exploitation. Stage 1 provides the system with visibility, Stage 2 provides structured cybersecurity intelligence and labelled attack experience, and Stage 3 uses these outputs to create an intelligent autonomous security-testing agent.

Safety and operational constraints will remain fundamental to the autonomous architecture. The SLM's decisions will be bounded by explicit policies governing allowable targets, tools, attack techniques, system states, risk thresholds, and physical consequences.

**Stage 3 Outcomes and Deliverables:**

* An SLM specifically adapted for embedded and CPS penetration-testing workflows.
* A training dataset generated from Stage 1 observability data and Stage 2 penetration-testing trajectories.
* Integration of the SLM with the CKG for external cybersecurity knowledge retrieval and reasoning.
* Autonomous selection and orchestration of penetration-testing tools.
* Dynamic generation and modification of multi-stage attack strategies.
* Automated interpretation of tool outputs and runtime system responses.
* Feedback-driven decision making based on previous successful and unsuccessful actions.
* Context-aware attack-path prioritization based on exploitability, system state, operational risk, and potential impact.
* Policy-controlled autonomous testing suitable for test benches, digital twins, hardware-in-the-loop environments, and authorized embedded/CPS systems.
* Automated generation of findings, attack-chain explanations, observed impact, evidence, and recommended mitigation actions.
* A reusable autonomous penetration-testing framework capable of improving as additional system observations and testing outcomes are accumulated.

The overarching objective of this work is therefore to develop a progressive platform that evolves from **deep system observability to knowledge-assisted security automation and ultimately to autonomous intelligent penetration testing** for embedded and cyber-physical systems.

The proposed research progression can be summarized as:

**Stage 1: Performance Testing & Observability**
**↓**
Deep understanding of the embedded/CPS runtime environment and generation of structured telemetry.

**Stage 2: Automated Pentesting + CKG-Assisted Intelligence**
**↓**
Correlation of telemetry with cyber-threat intelligence, automated attack-path generation, tool orchestration, and generation of penetration-testing experience.

**Stage 3: SLM-Driven Autonomous Intelligent Pentesting**
**↓**
An adaptive security-testing agent that reasons over system state, cybersecurity knowledge, available tools, and previous attack outcomes to autonomously plan, execute, evaluate, and refine penetration-testing strategies.

The key research contribution is the creation of a unified closed-loop architecture connecting **System Observability → Cybersecurity Knowledge Representation → Attack-Path Reasoning → Automated Tool Execution → Runtime Impact Analysis → Learning → Autonomous Decision Making**. This enables security assessment to evolve from static vulnerability identification toward continuous, context-aware, and intelligent evaluation of complex embedded and cyber-physical systems.







# Intelligent Performance Monitoring and Autonomous Pentesting for Embedded & Cyber-Physical Systems

## Problem

Embedded and Cyber-Physical Systems (CPS) used in aerospace, automotive, industrial automation, robotics, energy, and critical infrastructure operate under strict requirements for **performance, determinism, reliability, safety, and security**. However, current monitoring and security-assessment approaches are largely fragmented.

Performance tools typically observe individual layers such as CPU, memory, network, kernel, or application behavior, while security tools independently identify vulnerabilities or execute predefined penetration-testing actions. There is limited capability to correlate **runtime behavior, system configuration, vulnerabilities, threat intelligence, attack paths, and cyber-physical impact** within a unified framework.

This creates a need for an integrated platform that can first understand how an embedded/CPS environment behaves, then use cybersecurity intelligence to assess its attack surface, and ultimately perform adaptive and autonomous security testing.

## Proposed Solution

Develop a three-stage R&D platform that progressively evolves from **deep system observability** to **knowledge-assisted automated penetration testing** and finally to **SLM-driven autonomous security assessment**.

**Performance & Observability → Cybersecurity Knowledge & Automated Pentesting → Autonomous Intelligent Pentesting**

---

## Stage 1 – Performance Testing & Observability

Develop a unified observability platform capable of monitoring the complete embedded/CPS runtime environment across:

* Kernel and operating system
* Applications and processes
* CPU, memory, I/O and scheduling
* Device drivers and hardware interfaces
* Network traffic and protocol behavior
* Sensors, actuators and CPS telemetry

Technologies such as **eBPF, Wireshark/TShark, system performance counters, OS tracing utilities, RTOS/Linux instrumentation, and protocol-specific analyzers** will be integrated. Support can include embedded and industrial protocols such as **CAN/CAN-FD, Ethernet, MQTT, Modbus and OPC UA**.

### Key Deliverables

* Unified performance and observability tool
* Cross-layer telemetry collection and correlation
* Performance baselining and anomaly detection
* Structured system-state representation
* APIs and datasets for downstream cybersecurity analysis

---

## Stage 2 – Automated Pentesting with CKG-Assisted Intelligence

Extend Stage 1 with a **Cybersecurity Knowledge Graph (CKG)** connecting the observed system environment with cybersecurity intelligence such as:

**CPE → CVE → CWE → CVSS → CAPEC → MITRE ATT&CK / ATT&CK for ICS**

The CKG will model relationships between assets, software/firmware, vulnerabilities, attack prerequisites, privileges, interfaces, attack techniques, system topology, and potential impact.

Security tools such as **MITRE CALDERA, Metasploit, vulnerability scanners, reconnaissance tools, protocol-testing utilities, and other security frameworks** will then be orchestrated automatically.

The CKG and live Stage 1 telemetry will assist in selecting relevant tests, constructing attack paths, validating prerequisites, observing attack effects, and determining subsequent actions.

**Operational Loop:**
**Discover → Correlate → Plan → Execute → Observe → Validate → Update**

### Key Deliverables

* CTI-driven Cybersecurity Knowledge Graph
* Automated asset-to-CPE/CVE/CWE/CVSS mapping
* Attack-graph and attack-path generation
* Automated CALDERA/Metasploit/tool orchestration
* Runtime validation using Stage 1 telemetry
* Structured attack-result and system-state datasets
* Automated evidence-based penetration-testing reports

---

## Stage 3 – Autonomous Intelligent Pentesting using SLM

Develop a **Small Language Model (SLM)-based autonomous pentesting agent** using the knowledge and datasets generated during Stages 1 and 2.

The SLM will operate together with the **CKG, observability platform, penetration-testing tools, and historical attack outcomes** rather than relying solely on its internal model knowledge.

It will autonomously:

* Analyze the current system state
* Query relevant cybersecurity knowledge
* Identify candidate vulnerabilities and attack paths
* Select appropriate security tools
* Construct multi-stage attack strategies
* Execute authorized security tests
* Interpret tool and telemetry outputs
* Learn from successful and unsuccessful actions
* Dynamically modify the testing strategy
* Generate findings, evidence and mitigation recommendations

Explicit safety and authorization policies will constrain autonomous actions, particularly for safety-critical CPS environments. High-risk testing can be restricted to **digital twins, simulations, test benches and Hardware-in-the-Loop environments**.

### Key Deliverables

* Embedded/CPS-focused SLM security agent
* Training/fine-tuning dataset derived from Stages 1 and 2
* SLM + CKG + tool orchestration architecture
* Adaptive multi-stage attack planning
* Autonomous feedback-driven pentesting
* Policy-controlled execution framework
* Automated attack-chain explanation and reporting

---

## Final Vision

The final outcome is a unified platform capable of progressing from **observing how an embedded or cyber-physical system operates to autonomously understanding how it can be attacked and validated securely**.

**Stage 1** provides **visibility and system understanding**.
**Stage 2** provides **cybersecurity context, attack intelligence and automated execution**.
**Stage 3** provides **adaptive reasoning and autonomous decision-making**.

The resulting architecture creates a closed-loop security assessment framework:

**System Observability → CTI/CKG Intelligence → Attack-Path Reasoning → Automated Tool Execution → Runtime Impact Analysis → Learning → Autonomous Pentesting**

The long-term objective is to enable **continuous, context-aware and intelligent security validation of embedded and cyber-physical systems**, reducing manual penetration-testing effort while improving coverage, reproducibility, attack-path discovery and understanding of cyber-physical impact.
