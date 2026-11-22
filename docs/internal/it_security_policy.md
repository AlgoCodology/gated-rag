# IT Security Policy (Staff Only)

Classification: INTERNAL. Owner: Chief Information Security Officer (CISO), Dennis Vos. Version 4.2, effective 1 July 2025.

## 1. Data classification

| Level | Examples | Allowed storage |
|:---|:---|:---|
| Public | Course catalogue, press releases | Anywhere |
| Internal | Meeting minutes, staff directories | HomeStore, university Microsoft 365 |
| Confidential | Student grades, HR files, contracts | HomeStore encrypted volumes only |
| Strictly confidential | Medical data, exam papers before the exam, board strategy | Encrypted volumes with named access list |

## 2. Accounts and authentication

- Passwords must have at least **14 characters**. Staff passwords expire every **180 days**.
- **Multi-factor authentication (MFA)** is mandatory for all staff accounts. The preferred second factor is the HomeID Authenticator app. SMS codes are only allowed as a fallback.
- Accounts are locked after **5 failed login attempts** within 10 minutes. The IT Service Desk can unlock an account after verifying identity in person or via a video call.
- Shared accounts are prohibited, except for the generic exam-room account, which is reset after every exam session.

## 3. Incident reporting

Any suspected security incident (phishing click, lost laptop, malware, unauthorised access) must be reported to the IT Service Desk **within 1 hour** on internal extension **4444**, or by e-mail to security@homeuniversity.example. Lost or stolen devices are remotely wiped. Incidents that involve personal data must be reported to the Data Protection Officer, who notifies the Dutch Data Protection Authority within 72 hours when required.

## 4. Network

- The staff network (*HomeNet*) is separate from student and guest networks. Guests use *HomeGuest*, with a voucher valid for 24 hours.
- Remote access to internal systems requires the university VPN. The VPN gateway is `vpn.homeuniversity.example`.
- The server room in Building A (room A0.01) is accessible only with the red staff card and PIN. The maintenance code for the server room door is rotated monthly by Facilities.

## 5. Backup and recovery

HomeStore is backed up daily at 02:00 with 35 days of retention. The recovery time objective for the learning platform is 4 hours, and for the student information system 8 hours. A restore test is performed every quarter.

## 6. Generative AI

Staff may use the university's local AI assistant for internal and public data. Confidential and strictly confidential data may only be processed by models running on university infrastructure. Uploading such data to external AI services is prohibited.
