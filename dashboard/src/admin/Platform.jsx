import Overview from "../Overview.jsx";

// The numbers of the whole platform. The data comes from /api/admin/stats, which only admins can read.
export default function Platform() {
  return <Overview base="/api/admin/stats" title="Platform overview" scope="platform" />;
}