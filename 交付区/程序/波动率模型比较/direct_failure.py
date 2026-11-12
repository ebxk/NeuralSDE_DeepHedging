"""Recognize recorded financial rejections without accepting software failures."""

def known_direct_rejection(record,end,model):
    return (model=='localvol' and record.get('model')==model and record.get('end')==end
        and record.get('status')=='numerical_failed' and record.get('error_type')=='ValueError'
        and bool(record.get('parents')) and bool(record.get('sources'))
        and record.get('error','').startswith(('Dupire invalid:','Dupire undefined:',
            'current-day localvol conditional PDE failed')))
