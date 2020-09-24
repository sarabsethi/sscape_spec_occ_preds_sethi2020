import requests
import sqlite3
from collections import Counter

def init_sql_conn(sqlite_path):
    try:
        conn = sqlite3.connect(sqlite_path)
        _ = conn.execute('select count(*) from backbone;')
    except sqlite3.OperationalError:
        LOGGER.critical('Local GBIF database does not contain the backbone table')
        raise IOError('Local GBIF database does not contain the backbone table')
    except sqlite3.DatabaseError:
        LOGGER.critical('Local SQLite database not valid')
        raise IOError('Local SQLite database not valid')
    else:
        conn.row_factory = sqlite3.Row
        print('Successfully initialised SQL conn')
        return conn

def web_gbif_validate(tax, rnk, gbif_id=None):
    """
    Validates a taxon name and rank against the GBIF web API. It uses the API endpoint
    species/match?name=XXX&rank=YYY&strict=true
    endpoint to
    # safe to assume that the next least nested taxonomic level is the parent.
    # So, populate GBIF ID using species/match and then use species/{id} to populate the
    # return values
    Args:
        tax (str, None): The (case insensitive) taxon name
        rnk (str, None): The taxon rank
        gbif_id (int): Optional GBIF taxon id to resolve ambiguous taxon/rank
            combinations. If gbif_id is not None, then tax and rank can be
            None and the GBIF response will be used to populate those values.
    Returns:
        A dictionary with the following possible keys:
            'status': the outcome of the lookup with one of the following values:
                found, no_match, validation_fail, unknown_id, id_mismatch
            'user': an index list of the user provided information for synonymous usages
            'canon': an index list of the canonical information
            'hier': a list of 2-tuples of rank and GBIF ID for the taxonomic hierarchy
            'note': a string of any extra information provided by the search
        An index listhas the structure: (taxon ID, parent ID, name, rank, taxonomic status)
    """

    if gbif_id is None:
        # If no ID is provided then use the species/match?name={} endpoint to
        # try and find an ID for the combination
        url = u"http://api.gbif.org/v1/species/match?name={}&rank={}&strict=true".format(tax, rnk)
        tax_gbif = requests.get(url)

        # failures here suggest some kind of gateway problem
        if tax_gbif.status_code != 200:
            return {'status': 'validation_fail'}

        resp = tax_gbif.json()

        # check the response status
        if resp['matchType'] == u'NONE':
            # No match found - look for explanatory notes
            if 'note' in resp:
                return {'status': 'no_match', 'note': resp['note']}
            else:
                return {'status': 'no_match'}
        else:
            gbif_id = resp['usageKey']

    # Now use the species/{id} endpoint
    url = u"http://api.gbif.org/v1/species/{}".format(gbif_id)
    tax_gbif = requests.get(url)

    # unknown ID numbers return a 404 error
    if tax_gbif.status_code == 404:
        return {'status': 'unknown_id'}
    elif tax_gbif.status_code != 200:
        return {'status': 'validation_fail'}

    resp = tax_gbif.json()

    # If they were provided, check tax and rnk are compatible with the ID number
    if ((tax is not None and tax != resp['canonicalName']) or
            (rnk is not None and rnk.lower() != resp['rank'].lower())):
        return {'status': 'id_mismatch'}

    # Now we should have a single usage in resp so populate the return dictionaries and
    # set the response to be used for the hierarchy.

    # First, fill in the parent key if it is missing, which it is for Kingdom level taxa
    if resp['rank'] == 'KINGDOM':
        resp['parentKey'] = None

    if resp['taxonomicStatus'].lower() in ('accepted', 'doubtful'):
        # populate the return values
        ret = {'status': 'found',
               'canon': [resp['key'], resp['parentKey'], resp['canonicalName'],
                         resp['rank'].lower(), resp['taxonomicStatus'].lower()]}
        hier_resp = resp
    else:
        # look up the accepted usage
        usage_url = u"http://api.gbif.org/v1/species/{}".format(resp['acceptedKey'])
        accept = requests.get(usage_url)

        if accept.status_code != 200:
            return {'status': 'validation_fail'}
        else:
            acpt = accept.json()

        # populate the return values
        ret = {'status': 'found',
               'user': [resp['key'], resp['parentKey'], resp['canonicalName'],
                        resp['rank'].lower(), resp['taxonomicStatus'].lower()],
               'canon': [acpt['key'], acpt['parentKey'], acpt['canonicalName'],
                         acpt['rank'].lower(), acpt['taxonomicStatus'].lower()]}

        hier_resp = acpt

    # Add the taxonomic hierarchy from the accepted usage - these are tuples
    # to be used to extend a set for the taxonomic hierarchy
    ret['hier'] = [(rk, hier_resp[ky])
                   for rk, ky in [('kingdom', 'kingdomKey'), ('phylum', 'phylumKey'),
                                  ('class', 'classKey'), ('order', 'orderKey'),
                                  ('family', 'familyKey'), ('genus', 'genusKey'),
                                  ('species', 'speciesKey')]
                   if ky in hier_resp]

    # return the details
    return ret


def local_gbif_validate(conn, tax, rnk, gbif_id=None):
    """
    Validates a taxon name and rank against a connection to a local GBIF database.
    Args:
        conn (sqlite3.Connection): An sqlite3 connection to the backbone database
        tax (str, None): The (case sensitive) taxon name
        rnk (str, None): The taxon rank
        gbif_id (int): Optional GBIF taxon id to resolve ambiguous taxon/rank
            combinations. If gbif_id is not None, then tax and rank can be
            None and the GBIF response will be used to populate those values.
    Returns:
        A dictionary with the following possible keys:
            'status': the outcome of the lookup with one of the following values:
                found, no_match, validation_fail, unknown_id, id_mismatch
            'user': an index list of the user provided information for synonymous usages
            'canon': an index list of the canonical information
            'hier': a list of 2-tuples of rank and GBIF ID for the taxonomic hierarchy
            'note': a string of any extra information provided by the search
        An index list has the structure: (taxon ID, parent ID, name, rank, taxonomic status)
    """

    # Make sure the connection is returning results as sqlite.Row objects
    if conn.row_factory != sqlite3.Row:
        conn.row_factory = sqlite3.Row

    # This if else section either runs through with a single Row assigned
    # to tax_gbif or exits returning an error condition of some sort
    if gbif_id is not None:
        # get the record associated with the provided ID
        tax_sql = (u"select * from backbone where id = {}".format(gbif_id))
        tax_gbif = conn.execute(tax_sql).fetchone()

        # check there is a result and that it is congruent with any
        # provided taxon or rank information
        if tax_gbif is None:
            return {'status': 'unknown_id'}
        elif ((tax is not None and tax_gbif['canonical_name'] != tax) or
              (rnk is not None and tax_gbif['rank'].lower() != rnk.lower())):
            return {'status': 'id_mismatch'}

    else:
        # get the set of records associated with the taxon or rank
        tax_sql = (u"select * from backbone where canonical_name ='{}' and "
                    "rank= '{}';".format(tax, rnk.upper()))
        tax_gbif = conn.execute(tax_sql).fetchall()

        if len(tax_gbif) == 0:
            # No matching rows
            return {'status': 'no_match'}
        elif len(tax_gbif) == 1:
            # one matching row - extract it from the list
            tax_gbif = tax_gbif[0]
        else:
            # More than one row - try to mimic the preferred hits reported
            # by the GBIF API to select a single hit by looking at the counts
            # of the different statuses.

            # First, get the taxon statuses
            tx_status = [tx['status'].lower() for tx in tax_gbif]
            tx_counts = Counter(tx_status)

            if 'accepted' in tx_counts.keys():
                # Accepted hits are first preference
                if tx_counts['accepted'] == 1:
                    # only one accepted match alongside other usages so extract that hit
                    tax_gbif = tax_gbif[tx_status.index('accepted')]
                else:
                    # more than one accepted match, so return no match and a note, as
                    # the API interface does.
                    return {'status': 'no_match',
                            'note': 'Multiple equal matches for {}'.format(tax)}

            elif 'doubtful' in tx_counts.keys():
                # Doubtful hits get next preference - not quite sure about this!
                if tx_counts['doubtful'] == 1:
                    # only one doubtful match alongside other usages so extract that hit
                    tax_gbif = tax_gbif[tx_status.index('doubtful')]
                else:
                    # more than one doubtful match, so return no match and a note, as
                    # the API interface does.
                    return {'status': 'no_match',
                            'note': 'Multiple equal matches for {}'.format(tax)}

            else:
                # Rows now contain only synonyms (of varying kinds) and misapplied. Both of
                # these types have accepted usage values, so look for a unique accepted usage,
                # trapping the edge case of kingdoms, which have no parent_key.

                tx_acc = {tx['parent_key'] for tx in tax_gbif if tx['parent_key'] is not None}

                if len(tx_acc) > 1:
                    # More than one accepted usage
                    return {'status': 'no_match',
                            'note': 'Multiple equal matches for {}'.format(tax)}
                else:
                    # A single accepted usage - pick the first row to index
                    tax_gbif = tax_gbif[0]

    # Should now have a single row for the preferred hit, so package that up and set
    # what is going to be used to build the hierarchy
    if tax_gbif['status'].lower() in ['accepted', 'doubtful']:
        ret = {'status': 'found',
               'canon': [tax_gbif['id'], tax_gbif['parent_key'], tax_gbif['canonical_name'],
                         tax_gbif['rank'].lower(), tax_gbif['status'].lower()]}
        hier_row = tax_gbif
    else:
        # Look up the parent_key, which is the accepted usage key for synonyms.
        acc_sql = 'select * from backbone where id = {};'.format(tax_gbif['parent_key'])
        acc_gbif = conn.execute(acc_sql).fetchone()
        # fill in the return. The use of the accepted taxon parent key for the user entry
        # is deliberate: it points up the hierarchy not to the accepted taxon.
        ret = {'status': 'found',
               'canon': [acc_gbif['id'], acc_gbif['parent_key'], acc_gbif['canonical_name'],
                         acc_gbif['rank'].lower(), acc_gbif['status'].lower()],
               'user': [tax_gbif['id'], acc_gbif['parent_key'], tax_gbif['canonical_name'],
                        tax_gbif['rank'].lower(), tax_gbif['status'].lower()]}
        hier_row = acc_gbif

    # Add the taxonomic hierarchy from the preferred usage
    ret['hier'] = [(rk, hier_row[ky])
                   for rk, ky in [('kingdom', 'kingdom_key'), ('phylum', 'phylum_key'),
                                  ('class', 'class_key'), ('order', 'order_key'),
                                  ('family', 'family_key'), ('genus', 'genus_key'),
                                  ('species', 'species_key')]
                   if hier_row[ky] is not None]

    return ret
