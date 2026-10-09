"""enterprise-finance-mcp: one local MCP server standing in for the ERP, procurement,
vendor-management, HR and payments systems. Mock data lives in data/enterprise.json.

Run:  python -m enterprise_finance_mcp.server   (stdio transport)
"""
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

DATA_PATH = Path(os.environ.get("ENTERPRISE_DATA", Path(__file__).resolve().parents[1] / "data" / "enterprise.json"))
DATA: dict[str, Any] = json.loads(DATA_PATH.read_text())

mcp = MCPServer(
    "enterprise-finance-mcp",
    instructions="Read-only access to mock enterprise finance systems: invoices, POs, vendors, "
                 "bank accounts, vendor change history, approval policies, employees and payments.",
)


def _not_found(kind: str, key: str) -> dict[str, Any]:
    return {"found": False, "error": f"No {kind} with id '{key}'"}


@mcp.tool()
def get_invoice(invoice_id: str) -> dict[str, Any]:
    """Get an invoice: vendor, PO reference, amount, dates, line items and the remit-to bank details on the invoice."""
    inv = DATA["invoices"].get(invoice_id)
    return {"found": True, **inv} if inv else _not_found("invoice", invoice_id)


@mcp.tool()
def get_purchase_order(po_id: str) -> dict[str, Any]:
    """Get a purchase order: vendor, status, authorized amount, amount already invoiced, approver and any amendments."""
    po = DATA["purchase_orders"].get(po_id)
    return {"found": True, **po} if po else _not_found("purchase order", po_id)


@mcp.tool()
def get_vendor(vendor_id: str) -> dict[str, Any]:
    """Get a vendor master record: status, approval, onboarding date, category and risk tier."""
    v = DATA["vendors"].get(vendor_id)
    return {"found": True, **v} if v else _not_found("vendor", vendor_id)


@mcp.tool()
def get_vendor_bank_accounts(vendor_id: str) -> dict[str, Any]:
    """List all bank accounts on file for a vendor, including inactive ones, with when each was added and whether it is verified."""
    if vendor_id not in DATA["vendors"]:
        return _not_found("vendor", vendor_id)
    return {"found": True, "vendor_id": vendor_id, "accounts": DATA["vendor_bank_accounts"].get(vendor_id, [])}


@mcp.tool()
def get_vendor_change_history(vendor_id: str) -> dict[str, Any]:
    """Audit log of changes to a vendor's master data (bank account, remit email, address): who changed what, when, how it was requested, and secondary-approval status."""
    if vendor_id not in DATA["vendors"]:
        return _not_found("vendor", vendor_id)
    return {"found": True, "vendor_id": vendor_id, "changes": DATA["vendor_change_history"].get(vendor_id, [])}


@mcp.tool()
def get_approval_policy(policy_type: str) -> dict[str, Any]:
    """Get an approval policy. Known types: invoice_auto_approval, vendor_bank_change, po_variance, payment_release."""
    p = DATA["approval_policies"].get(policy_type)
    if not p:
        return {"found": False, "error": f"Unknown policy_type '{policy_type}'",
                "known_policy_types": sorted(DATA["approval_policies"])}
    return {"found": True, **p}


@mcp.tool()
def get_employee(employee_id: str) -> dict[str, Any]:
    """Get an employee record: name, role, department, status."""
    e = DATA["employees"].get(employee_id)
    return {"found": True, **e} if e else _not_found("employee", employee_id)


@mcp.tool()
def get_payment_history(vendor_id: str) -> dict[str, Any]:
    """Past payments to a vendor: dates, amounts and the account (last 4 digits) each was paid to."""
    if vendor_id not in DATA["vendors"]:
        return _not_found("vendor", vendor_id)
    return {"found": True, "vendor_id": vendor_id, "today": DATA["today"],
            "payments": DATA["payment_history"].get(vendor_id, [])}


if __name__ == "__main__":
    mcp.run("stdio")
