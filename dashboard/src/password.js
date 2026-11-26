const LABELS = ["too short", "weak", "fair", "good", "strong"];

// What the server requires is only the length. These are hints for the person who is typing.
export function passwordProblems(password) {
  const problems = [];
  if (password.length < 10) problems.push("use at least 10 characters");
  if (password.length > 1 && /^(.)\1+$/.test(password)) problems.push("do not repeat one character");
  return problems;
}

export function passwordStrength(password) {
  if (password.length === 0) return { score: 0, label: "" };
  if (password.length < 10) return { score: 0, label: LABELS[0] };
  let score = 1;
  if (password.length >= 14) score += 1;
  if (/[a-z]/i.test(password) && /\d/.test(password)) score += 1;
  if (/\s/.test(password) || /[^A-Za-z0-9]/.test(password) || password.length >= 20) score += 1;
  return { score, label: LABELS[score] };
}